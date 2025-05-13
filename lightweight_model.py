import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    BatchNormalization, Activation, SpatialDropout2D, Dropout,
    add, multiply
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#--------------------------------------
# 1) CONFIGURE GPU / CPU
#--------------------------------------
def configure_for_performance():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)

#--------------------------------------
# 2) CUSTOM LOSSES & METRICS
#--------------------------------------
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1. - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """ 0.7 * CCE + 0.3 * Dice, then mean over batch """
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return K.mean(0.7 * ce + 0.3 * dl)

def mean_iou_onehot(y_true, y_pred):
    """Convert one-hot to int labels, build confusion matrix, compute mean IoU."""
    # 1) Argmax to get [batch, H, W]
    y_true_cls = K.argmax(y_true, axis=-1)
    y_pred_cls = K.argmax(y_pred, axis=-1)
    # 2) Flatten
    y_true_f = K.flatten(y_true_cls)
    y_pred_f = K.flatten(y_pred_cls)
    # 3) Confusion matrix
    cm = tf.math.confusion_matrix(
        y_true_f, y_pred_f, num_classes=3, dtype=tf.float32
    )
    # 4) IoU per class
    diag = tf.linalg.diag_part(cm)
    rowsum = tf.reduce_sum(cm, axis=1)
    colsum = tf.reduce_sum(cm, axis=0)
    union = rowsum + colsum - diag
    iou = (diag + 1e-10) / (union + 1e-10)
    return K.mean(iou)

#--------------------------------------
# 3) DATA PREPROCESSING & AUGMENTATION
#--------------------------------------
def normalize_image(img, low_p=1, high_p=99):
    if img.min() == img.max():
        return img
    lo = np.percentile(img, low_p)
    hi = np.percentile(img, high_p)
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo + 1e-10)

def apply_bilateral_filter(nir, d=9, sc=75, ss=75):
    u8 = (np.clip(nir,0,1)*255).astype(np.uint8)
    f = cv2.bilateralFilter(u8, d, sc, ss)
    return f.astype(np.float32)/255.0

def calculate_ndvi(nir, red):
    eps = 1e-10
    ndvi = (nir - red) / (nir + red + eps)
    return (ndvi + 1)/2

def random_brightness_contrast(img, br=(-0.15,0.15), cr=(-0.15,0.15)):
    b = 1 + np.random.uniform(*br)
    img = np.clip(img*b,0,1)
    c = 1 + np.random.uniform(*cr)
    img = np.clip((img-0.5)*c+0.5,0,1)
    return img

def add_gaussian_noise(img, mean=0, std=0.01):
    return np.clip(img + np.random.normal(mean,std,img.shape),0,1)

def random_rotation(img, mask, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    # image
    out_i = np.stack([
        cv2.warpAffine(img[:,:,c], M, (w,h),
                       flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
        for c in range(img.shape[2])
    ], axis=-1)
    # mask (nearest)
    out_m = cv2.warpAffine(mask, M, (w,h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT)
    return out_i, out_m

def random_zoom(img, mask, zr=(-0.1,0.1)):
    h,w = img.shape[:2]
    z = 1 + np.random.uniform(*zr)
    nh,nw = int(h*z), int(w*z)
    if z>1:
        # zoom in then crop
        tmp_i = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
        tmp_m = cv2.resize(mask, (nw,nh), interpolation=cv2.INTER_NEAREST)
        sh,sw = (nh-h)//2, (nw-w)//2
        return tmp_i[sh:sh+h,sw:sw+w], tmp_m[sh:sh+h,sw:sw+w]
    else:
        # zoom out then pad
        tmp_i = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
        tmp_m = cv2.resize(mask, (nw,nh), interpolation=cv2.INTER_NEAREST)
        pad_h, pad_w = (h-nh)//2, (w-nw)//2
        out_i = np.zeros_like(img); out_m = np.zeros_like(mask)
        out_i[pad_h:pad_h+nh, pad_w:pad_w+nw] = tmp_i
        out_m[pad_h:pad_h+nh, pad_w:pad_w+nw] = tmp_m
        return out_i, out_m

def load_and_preprocess(rgb_dir, nir_dir, mask_dir, img_size=(128,128), max_images=None):
    X, Y = [], []
    files = sorted(os.listdir(rgb_dir))
    if max_images: files = files[:max_images]
    for fn in files:
        rgb_p = os.path.join(rgb_dir, fn)
        nir_p = os.path.join(nir_dir, fn)
        m_p   = os.path.join(mask_dir, fn)
        if not (os.path.exists(nir_p) and os.path.exists(m_p)): continue
        # load
        rgb = cv2.cvtColor(cv2.imread(rgb_p), cv2.COLOR_BGR2RGB)
        nir = cv2.imread(nir_p, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
        if rgb is None or nir is None or msk is None: continue
        # resize
        rgb = cv2.resize(rgb, img_size)
        nir = cv2.resize(nir, img_size)
        msk = cv2.resize(msk, img_size, interpolation=cv2.INTER_NEAREST)
        # normalize
        rgb = rgb.astype(np.float32)/255.0
        nir = nir.astype(np.float32)/255.0
        for c in range(3): rgb[:,:,c] = normalize_image(rgb[:,:,c])
        nir = normalize_image(nir)
        # build 3-channel input: G, filtered NIR, NDVI
        G = rgb[:,:,1]
        fnir = apply_bilateral_filter(nir)
        ndvi = calculate_ndvi(nir, rgb[:,:,0])
        x = np.stack([G, fnir, ndvi], axis=-1)
        # one-hot mask
        y = np.zeros((*img_size,3),dtype=np.float32)
        for c in range(3):
            y[:,:,c] = (msk==c).astype(np.float32)
        X.append(x); Y.append(y)
    X = np.array(X); Y = np.array(Y)
    if len(X)==0:
        raise RuntimeError("No images found. Check paths.")
    return X, Y

def augment_data(X, Y, factor=2.5):
    out_X, out_Y = list(X), list(Y)
    if factor<=1: return np.array(out_X), np.array(out_Y)
    total = int((factor-1)*len(X))
    per   = max(1, total//len(X))
    for i in range(len(X)):
        img, msk = X[i], np.argmax(Y[i],axis=-1)
        for _ in range(per):
            # flip/rotate/zoom/brightness/noise
            if np.random.rand()<0.5:
                img2, m2 = np.fliplr(img), np.fliplr(msk)
            else:
                img2, m2 = img.copy(), msk.copy()
            if np.random.rand()<0.3:
                img2, m2 = random_rotation(img2,m2, max_angle=10)
            if np.random.rand()<0.3:
                img2, m2 = random_zoom(img2,m2, zr=(-0.08,0.08))
            if np.random.rand()<0.5:
                img2 = random_brightness_contrast(img2, br=(-0.1,0.1), cr=(-0.1,0.1))
            if np.random.rand()<0.2:
                img2 = add_gaussian_noise(img2, std=0.005)
            # re–one-hot
            y2 = np.zeros_like(Y[i])
            for c in range(3): y2[:,:,c] = (m2==c)
            out_X.append(img2); out_Y.append(y2)
    return np.array(out_X), np.array(out_Y)

#--------------------------------------
# 4) BUILD MODEL
#--------------------------------------
def attention_gate(x, g, f):
    θx = Conv2D(f,1,padding='same')(x)
    φg = Conv2D(f,1,padding='same')(g)
    h  = Activation('relu')(add([θx,φg]))
    ψ  = Conv2D(1,1,padding='same')(h)
    α  = Activation('sigmoid')(ψ)
    return multiply([x, α])

def build_unet(input_shape=(128,128,3), n_classes=3):
    inp = Input(shape=input_shape)
    reg = tf.keras.regularizers.l2(5e-5)
    # Encoder
    def conv_block(x, filters, dp):
        x = Conv2D(filters,3,padding='same',kernel_regularizer=reg)(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters,3,padding='same',kernel_regularizer=reg)(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = Activation('relu')(x)
        p = MaxPooling2D()(x)
        p = SpatialDropout2D(dp)(p)
        return x, p

    c1, p1 = conv_block(inp, 32, 0.1)
    c2, p2 = conv_block(p1, 64, 0.2)
    c3, p3 = conv_block(p2,128, 0.3)

    # Bridge
    b  = Conv2D(256,3,padding='same',kernel_regularizer=reg)(p3)
    b  = BatchNormalization(momentum=0.95)(b)
    b  = Activation('relu')(b)
    b  = Conv2D(256,3,padding='same',kernel_regularizer=reg)(b)
    b  = BatchNormalization(momentum=0.95)(b)
    b  = Activation('relu')(b)
    b  = Dropout(0.35)(b)

    # Decoder
    def up_block(x, skip, filters, dp):
        u = UpSampling2D()(x)
        a = attention_gate(skip, u, filters)
        m = concatenate([u, a])
        m = Conv2D(filters,3,padding='same',kernel_regularizer=reg)(m)
        m = BatchNormalization(momentum=0.95)(m)
        m = Activation('relu')(m)
        m = Conv2D(filters,3,padding='same',kernel_regularizer=reg)(m)
        m = BatchNormalization(momentum=0.95)(m)
        m = Activation('relu')(m)
        m = SpatialDropout2D(dp)(m)
        return m

    d1 = up_block(b, c3, 128, 0.2)
    d2 = up_block(d1, c2, 64,  0.1)
    d3 = up_block(d2, c1, 32,  0.0)

    out = Conv2D(n_classes, 1, activation='softmax')(d3)
    return Model(inputs=inp, outputs=out)

#--------------------------------------
# 5) CRF POST-PROCESSING
#--------------------------------------
def apply_crf_postprocessing(image, probs, num_classes=3,
                              theta_alpha=40, theta_beta=10, iters=3):
    refined = probs.copy()
    if image.max()>1.0: image = image/255.0
    img_u8 = (image*255).astype(np.uint8)
    for _ in range(iters):
        for c in range(num_classes):
            pc = (refined[:,:,c]*255).astype(np.uint8)
            f  = cv2.bilateralFilter(pc, 5, theta_alpha, theta_beta)
            refined[:,:,c] = f/255.0
        summed = refined.sum(axis=-1,keepdims=True)
        refined /= (summed + 1e-10)
    return np.argmax(refined, axis=-1)

#--------------------------------------
# 6) TRAIN & EVALUATE
#--------------------------------------
def train_model(X_train, Y_train, X_val, Y_val,
                batch_size=8, epochs=40):
    model = build_unet(input_shape=X_train.shape[1:])
    opt   = Adam(learning_rate=5e-4, clipnorm=1.0, clipvalue=0.5)
    model.compile(
        optimizer=opt,
        loss=combined_loss,
        metrics=['accuracy', mean_iou_onehot]
    )

    callbacks = [
        EarlyStopping('val_mean_iou_onehot', mode='max',
                      patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau('val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_model.h5',
                        monitor='val_mean_iou_onehot',
                        mode='max', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

def plot_history(h):
    plt.figure(figsize=(18,5))
    # Loss
    plt.subplot(1,3,1)
    plt.plot(h.history['loss'], label='train')
    plt.plot(h.history['val_loss'], label='val')
    plt.title('Loss'); plt.legend(); plt.grid(True)
    # Acc
    plt.subplot(1,3,2)
    plt.plot(h.history['accuracy'], label='train')
    plt.plot(h.history['val_accuracy'], label='val')
    plt.title('Accuracy'); plt.legend(); plt.grid(True)
    # IoU
    plt.subplot(1,3,3)
    plt.plot(h.history['mean_iou_onehot'], label='train')
    plt.plot(h.history['val_mean_iou_onehot'], label='val')
    plt.title('Mean IoU'); plt.legend(); plt.grid(True)
    plt.show()

def visualize_samples(X, Y, model, n=5, apply_crf=False):
    preds = model.predict(X[:n])
    if apply_crf:
        preds = np.stack([
            apply_crf_postprocessing(X[i], preds[i])
            for i in range(n)
        ],axis=0)
    else:
        preds = np.argmax(preds, axis=-1)

    truths = np.argmax(Y[:n], axis=-1)
    colors = np.array([[0,0,0],[0,255,0],[255,0,0]])
    for i in range(n):
        img = X[i][...,:3]
        gt  = colors[truths[i]]
        pr  = colors[preds[i]]
        fig,axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].imshow(img); axs[0].set_title("Input"); axs[0].axis('off')
        axs[1].imshow(gt);  axs[1].set_title("GT");    axs[1].axis('off')
        axs[2].imshow(pr);  axs[2].set_title("Pred");  axs[2].axis('off')
        plt.show()

#--------------------------------------
# 7) MAIN PIPELINE
#--------------------------------------
if __name__ == "__main__":
    configure_for_performance()

    RGB_DIR = "/Users/varad/Documents/RESEARCH-PROJECT/jesi_06_13/rgb"
    NIR_DIR = "/Users/varad/Documents/RESEARCH-PROJECT/jesi_06_13/nir"
    MASK_DIR = "/Users/varad/Documents/RESEARCH-PROJECT/jesi_06_13/gt"

    X, Y = load_and_preprocess(RGB_DIR, NIR_DIR, MASK_DIR,
                               img_size=(128,128), max_images=None)

    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_te, Y_val, Y_te = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42)

    # optional augmentation
    X_tr, Y_tr = augment_data(X_tr, Y_tr, factor=2.0)

    model, hist = train_model(X_tr, Y_tr, X_val, Y_val,
                              batch_size=8, epochs=20)

    plot_history(hist)

    print("\n--- Validation Samples ---")
    visualize_samples(X_val, Y_val, model, n=5, apply_crf=False)
    print("\n--- CRF-refined Samples ---")
    visualize_samples(X_val, Y_val, model, n=5, apply_crf=True)

    # final test evaluation
    print("\n--- Final Test Metrics ---")
    res = model.evaluate(X_te, Y_te, verbose=1)
    print(dict(zip(model.metrics_names, res)))
