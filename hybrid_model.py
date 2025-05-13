import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, add, multiply, Lambda
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


# Configure TensorFlow for better performance
def configure_for_performance():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU acceleration enabled")
        except RuntimeError as e:
            print(e)
    else:
        print("Running on CPU")

    # Set thread count for CPU operations
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)


# Custom metrics and loss functions
def dice_coefficient(y_true, y_pred):
    """Calculate Dice coefficient for segmentation"""
    smooth = 1.0
    # Flatten the prediction and true values
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss for segmentation tasks"""
    return 1 - dice_coefficient(y_true, y_pred)


def weighted_categorical_crossentropy(weights):
    """Weighted cross-entropy loss function for imbalanced classes"""
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # Scale predictions so they sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # Clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Calculate weighted cross-entropy
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return K.mean(loss)

    return loss


# IoU metrics for multi-class segmentation
def mean_iou(y_true, y_pred):
    """Calculate mean IoU for multi-class segmentation"""
    num_classes = K.int_shape(y_pred)[-1]

    # Convert probabilities to class indices
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)

    # Reshape to 1D arrays
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    # Accumulate IoU for each class
    mean_iou = 0.0
    for i in range(num_classes):
        true_class = K.cast(K.equal(y_true, i), K.floatx())
        pred_class = K.cast(K.equal(y_pred, i), K.floatx())

        # Calculate intersection and union
        intersection = K.sum(true_class * pred_class)
        union = K.sum(true_class) + K.sum(pred_class) - intersection

        # Add to mean IoU with small epsilon to avoid division by zero
        iou = (intersection + K.epsilon()) / (union + K.epsilon())
        mean_iou += iou

    # Return average
    return mean_iou / K.cast(num_classes, K.floatx())


def class_iou(class_id):
    """Calculate IoU for a specific class"""

    def iou(y_true, y_pred):
        # Convert probabilities to class indices
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.argmax(y_true, axis=-1)

        # Create binary masks for this class
        true_class = K.cast(K.equal(y_true, class_id), K.floatx())
        pred_class = K.cast(K.equal(y_pred, class_id), K.floatx())

        # Calculate intersection and union
        intersection = K.sum(true_class * pred_class)
        union = K.sum(true_class) + K.sum(pred_class) - intersection

        return (intersection + K.epsilon()) / (union + K.epsilon())

    # Name the function for better logging
    iou.__name__ = f'iou_class_{class_id}'
    return iou


# Enhanced multispectral image processing functions
def apply_bilateral_filter(nir_img, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter to NIR image"""
    # Ensure image is properly scaled for bilateral filter (0-255)
    nir_scaled = np.clip(nir_img * 255, 0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(nir_scaled, d, sigma_color, sigma_space)
    return filtered / 255.0  # Return to 0-1 range


def calculate_ndvi(nir, red):
    """Calculate NDVI from NIR and Red channels"""
    # Avoid division by zero
    epsilon = 1e-10
    ndvi = (nir - red) / (nir + red + epsilon)
    # Normalize to 0-1 range
    ndvi = (ndvi + 1) / 2
    return ndvi


def normalize_image(img, lower_percent=1, upper_percent=99):
    """Enhance contrast by normalizing image between percentiles and stretching"""
    if img.min() == img.max():
        return img  # No normalization needed if image is constant

    low = np.percentile(img, lower_percent)
    high = np.percentile(img, upper_percent)

    # Clip and rescale to 0-1
    img_norm = np.clip(img, low, high)
    img_norm = (img_norm - low) / (high - low + 1e-10)

    return img_norm


# Load and preprocess multispectral images (RGB + NIR)
def load_and_preprocess_multispectral_images(rgb_dir, nir_dir, mask_dir, img_size=(128, 128), max_images=None):
    images_g_fnir_ndvi = []  # Green + Filtered NIR + NDVI
    masks = []

    # Check if directories exist
    for dir_path in [rgb_dir, nir_dir, mask_dir]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory not found: {dir_path}")

    # Get RGB files
    rgb_files = [f for f in sorted(os.listdir(rgb_dir))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if max_images:
        rgb_files = rgb_files[:max_images]

    print(f"Loading up to {len(rgb_files)} multispectral images")

    for rgb_name in rgb_files:
        # Construct paths
        rgb_path = os.path.join(rgb_dir, rgb_name)
        nir_name = rgb_name  # Assuming NIR has same filename

        # Find corresponding NIR file (with possibly different extension)
        nir_path = None
        nir_base = os.path.splitext(rgb_name)[0]
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            potential_nir_path = os.path.join(nir_dir, nir_base + ext)
            if os.path.exists(potential_nir_path):
                nir_path = potential_nir_path
                break

        if nir_path is None:
            print(f"Warning: No NIR image found for {rgb_name}, skipping")
            continue

        # Find corresponding mask file (with possibly different extension)
        mask_path = None
        mask_base = os.path.splitext(rgb_name)[0]
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            potential_mask_path = os.path.join(mask_dir, mask_base + ext)
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break

        if mask_path is None:
            print(f"Warning: No mask found for {rgb_name}, skipping")
            continue

        try:
            # Load RGB image
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is None:
                print(f"Warning: Could not read RGB image {rgb_path}, skipping")
                continue
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, img_size)

            # Load NIR image
            nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
            if nir_img is None:
                print(f"Warning: Could not read NIR image {nir_path}, skipping")
                continue
            nir_img = cv2.resize(nir_img, img_size)

            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}, skipping")
                continue
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

            # Normalize to 0-1 range
            rgb_img = rgb_img / 255.0
            nir_img = nir_img / 255.0

            # Apply contrast enhancement
            for i in range(3):
                rgb_img[:, :, i] = normalize_image(rgb_img[:, :, i])
            nir_img = normalize_image(nir_img)

            # Extract channels
            red_channel = rgb_img[:, :, 0]
            green_channel = rgb_img[:, :, 1]

            # Apply bilateral filter to NIR
            filtered_nir = apply_bilateral_filter(nir_img)

            # Calculate NDVI
            ndvi = calculate_ndvi(nir_img, red_channel)

            # Create G + Filtered-NIR + NDVI as suggested in the paper
            g_fnir_ndvi = np.dstack((green_channel, filtered_nir, ndvi))

            # Create one-hot encoded mask
            # Identify unique classes in the mask
            unique_classes = np.unique(mask)

            # Ensure we have exactly 3 classes (soil=0, crop=1, weed=2)
            if len(unique_classes) > 3:
                print(f"Warning: More than 3 classes in mask: {unique_classes}, will use first 3 classes only")
                unique_classes = unique_classes[:3]

            # Create one-hot encoded mask
            mask_one_hot = np.zeros((*img_size, 3), dtype=np.float32)
            for i, class_id in enumerate(range(3)):
                if class_id in unique_classes:
                    mask_one_hot[:, :, i] = (mask == class_id).astype(np.float32)

            # Check if mask contains any pixels for each class
            class_counts = [np.sum(mask_one_hot[:, :, i]) for i in range(3)]
            if min(class_counts) == 0:
                print(f"Warning: Mask for {rgb_name} is missing one or more classes: {class_counts}")
                # For debugging, we could skip these, but let's keep them for now

            # Append to lists
            images_g_fnir_ndvi.append(g_fnir_ndvi)
            masks.append(mask_one_hot)

        except Exception as e:
            print(f"Error processing {rgb_name}: {str(e)}")
            continue

    if not images_g_fnir_ndvi:
        raise ValueError("No valid images found. Check your image/mask directories and formats.")

    print(f"Successfully loaded {len(images_g_fnir_ndvi)} images")
    return np.array(images_g_fnir_ndvi), np.array(masks)


# Advanced data augmentation
def augment_data(images, masks, augmentation_factor=4):
    """Apply comprehensive data augmentation for small datasets"""
    if augmentation_factor <= 1:
        return images, masks

    augmented_images = []
    augmented_masks = []

    # Add original images
    augmented_images.extend(images)
    augmented_masks.extend(masks)

    num_original = len(images)
    num_to_generate = int(num_original * (augmentation_factor - 1))

    # Generate augmented images
    for _ in range(num_to_generate):
        idx = np.random.randint(0, num_original)
        img = images[idx].copy()
        mask = masks[idx].copy()

        # Apply random transformations
        # 1. Horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        # 2. Vertical flip
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)

        # 3. Random rotation (0, 90, 180, or 270 degrees)
        k = np.random.randint(0, 4)  # 0=0°, 1=90°, 2=180°, 3=270°
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)

        # 4. Random brightness variation
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness_factor, 0, 1)

        # 5. Random contrast variation
        if np.random.rand() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            img = np.clip((img - 0.5) * contrast_factor + 0.5, 0, 1)

        # 6. Random crop and resize (zoom)
        if np.random.rand() > 0.5:
            h, w = img.shape[0], img.shape[1]
            zoom_factor = np.random.uniform(0.8, 0.95)
            crop_h, crop_w = int(h * zoom_factor), int(w * zoom_factor)

            # Random crop position
            start_h = np.random.randint(0, h - crop_h + 1)
            start_w = np.random.randint(0, w - crop_w + 1)

            # Crop
            img_crop = img[start_h:start_h + crop_h, start_w:start_w + crop_w, :]
            mask_crop = mask[start_h:start_h + crop_h, start_w:start_w + crop_w, :]

            # Resize back to original size
            img = cv2.resize(img_crop, (w, h))

            # Use nearest neighbor for mask to preserve class labels
            mask_resized = np.zeros_like(mask)
            for c in range(mask.shape[2]):
                mask_resized[:, :, c] = cv2.resize(
                    mask_crop[:, :, c], (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
            mask = mask_resized

        # 7. Random noise addition
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)

        # 8. Random channel-wise adjustments
        if np.random.rand() > 0.7:
            for c in range(img.shape[2]):
                channel_factor = np.random.uniform(0.9, 1.1)
                img[:, :, c] = np.clip(img[:, :, c] * channel_factor, 0, 1)

        # Add the augmented pair to the lists
        augmented_images.append(img)
        augmented_masks.append(mask)

    print(f"Generated {len(augmented_images)} images after augmentation")
    return np.array(augmented_images), np.array(augmented_masks)


# U-Net model with ResNet50 encoder
def build_unet_resnet50(input_shape=(128, 128, 3), num_classes=3):
    """Build U-Net with ResNet50 encoder as mentioned in the paper"""
    # Input
    inputs = Input(shape=input_shape)

    # Use pre-trained ResNet50 as encoder
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Identify skip connections from ResNet50
    skip1 = resnet50.get_layer('conv1_relu').output  # 64x64
    skip2 = resnet50.get_layer('conv2_block3_out').output  # 32x32
    skip3 = resnet50.get_layer('conv3_block4_out').output  # 16x16
    skip4 = resnet50.get_layer('conv4_block6_out').output  # 8x8

    # Bottleneck
    bottleneck = resnet50.get_layer('conv5_block3_out').output  # 4x4

    # Decoder path
    # First upsampling block (4x4 -> 8x8)
    up1 = Conv2D(512, 3, padding='same', activation='relu')(bottleneck)
    up1 = BatchNormalization()(up1)
    up1 = UpSampling2D(size=(2, 2))(up1)
    # Connect with skip4
    up1 = concatenate([up1, skip4])
    up1 = Conv2D(512, 3, padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(0.2)(up1)

    # Second upsampling block (8x8 -> 16x16)
    up2 = Conv2D(256, 3, padding='same', activation='relu')(up1)
    up2 = BatchNormalization()(up2)
    up2 = UpSampling2D(size=(2, 2))(up2)
    # Connect with skip3
    up2 = concatenate([up2, skip3])
    up2 = Conv2D(256, 3, padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(0.2)(up2)

    # Third upsampling block (16x16 -> 32x32)
    up3 = Conv2D(128, 3, padding='same', activation='relu')(up2)
    up3 = BatchNormalization()(up3)
    up3 = UpSampling2D(size=(2, 2))(up3)
    # Connect with skip2
    up3 = concatenate([up3, skip2])
    up3 = Conv2D(128, 3, padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(0.2)(up3)

    # Fourth upsampling block (32x32 -> 64x64)
    up4 = Conv2D(64, 3, padding='same', activation='relu')(up3)
    up4 = BatchNormalization()(up4)
    up4 = UpSampling2D(size=(2, 2))(up4)
    # Connect with skip1
    up4 = concatenate([up4, skip1])
    up4 = Conv2D(64, 3, padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Dropout(0.2)(up4)

    # Final upsampling to original size (64x64 -> 128x128)
    up5 = Conv2D(32, 3, padding='same', activation='relu')(up4)
    up5 = BatchNormalization()(up5)
    up5 = UpSampling2D(size=(2, 2))(up5)

    # Output layer with softmax activation for multi-class segmentation
    outputs = Conv2D(num_classes, 1, activation='softmax')(up5)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model


# Apply CRF for post-processing
def apply_crf(image, prediction, num_classes=3):
    """Apply CRF to refine segmentation boundaries"""
    h, w = prediction.shape[:2]

    # Prepare input for CRF
    # If prediction is one-hot encoded, convert to class probabilities
    if len(prediction.shape) == 3 and prediction.shape[2] == num_classes:
        probs = prediction
    else:
        # Convert class indices to one-hot encoding
        probs = np.zeros((h, w, num_classes), dtype=np.float32)
        for i in range(num_classes):
            probs[:, :, i] = (prediction == i).astype(np.float32)

    # Create CRF model
    d = dcrf.DenseCRF2D(w, h, num_classes)

    # Unary potential based on class probabilities
    unary = unary_from_softmax(probs.transpose(2, 0, 1))
    d.setUnaryEnergy(unary)

    # Add pairwise potentials (position-dependent term)
    d.addPairwiseGaussian(sxy=3, compat=3)

    # Add appearance kernel (color-dependent term)
    # Convert image to uint8 if needed
    if image.max() <= 1.0:
        img_for_crf = (image * 255).astype(np.uint8)
    else:
        img_for_crf = image.astype(np.uint8)

    # Set parameters for appearance kernel - use smaller srgb for more sensitivity to color
    d.addPairwiseBilateral(sxy=50, srgb=10, rgbim=img_for_crf, compat=10)

    # Inference - 5 iterations
    Q = d.inference(5)

    # Get final map
    map_result = np.argmax(np.array(Q).reshape((num_classes, h, w)), axis=0)

    return map_result


# Training function with class weights for imbalanced data
def train_unet_model(X_train, Y_train, X_val, Y_val, img_size=(128, 128),
                     batch_size=8, epochs=50, class_weights=None):
    """Train U-Net with ResNet50 encoder and handle class imbalance"""
    # Build model
    model = build_unet_resnet50(input_shape=(img_size[0], img_size[1], 3), num_classes=3)

    # Set class weights if not provided
    if class_weights is None:
        # Count pixels of each class in training masks
        class_counts = np.sum(Y_train, axis=(0, 1, 2))
        total_pixels = np.sum(class_counts)

        # Calculate class weights (inverse frequency)
        class_weights = total_pixels / (class_counts * 3)  # multiply by num_classes
        class_weights = class_weights / np.sum(class_weights)  # normalize

        print(f"Automatic class weights: {class_weights}")

    # Define loss function with class weights
    loss_function = weighted_categorical_crossentropy(class_weights)

    # Create custom metrics for each class
    soil_iou = class_iou(0)  # soil
    crop_iou = class_iou(1)  # crop
    weed_iou = class_iou(2)  # weed

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=loss_function,
        metrics=['accuracy', mean_iou, soil_iou, crop_iou, weed_iou]
    )

    # Model summary
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
        ModelCheckpoint('best_unet_model.h5', monitor='val_mean_iou',
                        mode='max', save_best_only=True, verbose=1)
    ]

    # Train model
    print("Training U-Net model...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# Evaluate model performance with and without CRF
def evaluate_model(model, X_test, Y_test, apply_crf_postprocessing=True):
    """Evaluate model performance with and without CRF post-processing"""
    print("Evaluating model on test set...")

    # Get model predictions
    Y_pred = model.predict(X_test)

    # Convert predictions and ground truth to class indices
    Y_pred_classes = np.argmax(Y_pred, axis=-1)
    Y_true_classes = np.argmax(Y_test, axis=-1)

    # Calculate IoU metrics for each class
    class_names = ['Soil', 'Crop', 'Weed']
    ious_without_crf = []

    print("\nIoU scores without CRF:")
    for class_id in range(3):
        # Get binary masks for this class
        y_true = (Y_true_classes == class_id)
        y_pred = (Y_pred_classes == class_id)

        # Calculate IoU
        intersection = np.sum(y_true & y_pred)
        union = np.sum(y_true | y_pred)
        iou = intersection / (union + 1e-10)

        ious_without_crf.append(iou)
        print(f"{class_names[class_id]} IoU: {iou:.4f}")

    mean_iou_without_crf = np.mean(ious_without_crf)
    print(f"Mean IoU: {mean_iou_without_crf:.4f}")

    # Apply CRF post-processing if requested
    if apply_crf_postprocessing:
        print("\nApplying CRF post-processing...")
        refined_preds = []

        for i in range(len(X_test)):
            refined = apply_crf(X_test[i], Y_pred[i])
            refined_preds.append(refined)

        refined_preds = np.array(refined_preds)

        # Calculate IoU metrics with CRF
        ious_with_crf = []

        print("\nIoU scores with CRF:")
        for class_id in range(3):
            # Get binary masks for this class
            y_true = (Y_true_classes == class_id)
            y_pred = (refined_preds == class_id)

            # Calculate IoU
            intersection = np.sum(y_true & y_pred)
            union = np.sum(y_true | y_pred)
            iou = intersection / (union + 1e-10)

            ious_with_crf.append(iou)
            print(f"{class_names[class_id]} IoU: {iou:.4f}")

        mean_iou_with_crf = np.mean(ious_with_crf)
        print(f"Mean IoU: {mean_iou_with_crf:.4f}")

        # Return predictions for visualization
        return Y_pred, refined_preds, ious_without_crf, ious_with_crf

    return Y_pred, None, ious_without_crf, None


# Visualize segmentation results
def visualize_results(X_test, Y_test, Y_pred, refined_preds=None, num_samples=5):
    """Visualize segmentation results with and without CRF"""
    # Convert predictions to class indices if they are not already
    if Y_pred.shape[-1] > 1:  # One-hot encoded
        Y_pred_classes = np.argmax(Y_pred, axis=-1)
    else:
        Y_pred_classes = Y_pred

    # Convert ground truth to class indices if it's one-hot encoded
    if Y_test.shape[-1] > 1:  # One-hot encoded
        Y_true_classes = np.argmax(Y_test, axis=-1)
    else:
        Y_true_classes = Y_test

    # Class colors for visualization (RGB format)
    colors = [
        [0, 0, 0],  # Soil - Black
        [0, 255, 0],  # Crop - Green
        [255, 0, 0]  # Weed - Red
    ]
    colors = np.array(colors)

    # Limit number of samples to visualize
    num_samples = min(num_samples, len(X_test))

    # Create figure
    if refined_preds is not None:
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4 * num_samples))
        fig.suptitle('Segmentation Results with CRF Enhancement', fontsize=16)
    else:
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
        fig.suptitle('Segmentation Results', fontsize=16)

    for i in range(num_samples):
        # Handle case for single sample
        if num_samples == 1:
            ax_row = axes
        else:
            ax_row = axes[i]

        # Original image (use first 3 channels if more than 3)
        if X_test[i].shape[-1] >= 3:
            display_img = X_test[i][:, :, :3]  # Use first 3 channels
        else:
            # Create a 3-channel image from the first channel
            display_img = np.dstack([X_test[i][:, :, 0]] * 3)

        ax_row[0].imshow(display_img)
        ax_row[0].set_title('Input Image')
        ax_row[0].axis('off')

        # Ground truth mask
        true_mask_rgb = np.zeros((*Y_true_classes[i].shape, 3), dtype=np.uint8)
        for j in range(3):
            mask = Y_true_classes[i] == j
            true_mask_rgb[mask] = colors[j]

        ax_row[1].imshow(true_mask_rgb)
        ax_row[1].set_title('Ground Truth')
        ax_row[1].axis('off')

        # Model prediction without CRF
        pred_mask_rgb = np.zeros((*Y_pred_classes[i].shape, 3), dtype=np.uint8)
        for j in range(3):
            mask = Y_pred_classes[i] == j
            pred_mask_rgb[mask] = colors[j]

        ax_row[2].imshow(pred_mask_rgb)
        ax_row[2].set_title('Model Prediction')
        ax_row[2].axis('off')

        # CRF refined prediction if available
        if refined_preds is not None:
            refined_mask_rgb = np.zeros((*refined_preds[i].shape, 3), dtype=np.uint8)
            for j in range(3):
                mask = refined_preds[i] == j
                refined_mask_rgb[mask] = colors[j]

            ax_row[3].imshow(refined_mask_rgb)
            ax_row[3].set_title('CRF Refined')
            ax_row[3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
    plt.savefig('segmentation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# Plot training history
def plot_training_history(history):
    """Plot training history including loss and metrics"""
    plt.figure(figsize=(15, 10))

    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot training & validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot training & validation mean IoU
    plt.subplot(2, 2, 3)
    plt.plot(history.history['mean_iou'], label='Training Mean IoU')
    plt.plot(history.history['val_mean_iou'], label='Validation Mean IoU')
    plt.title('Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot training & validation class-wise IoU
    plt.subplot(2, 2, 4)

    # Check which class IoU metrics are available
    class_metrics = []
    for key in history.history:
        if 'iou_class_' in key and not key.startswith('val_'):
            class_metrics.append(key)

    for metric in class_metrics:
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')

    plt.title('Class-wise IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# Compare results with and without CRF
def plot_iou_comparison(ious_without_crf, ious_with_crf=None):
    """Plot IoU comparison with and without CRF"""
    class_names = ['Soil', 'Crop', 'Weed', 'Mean']

    # Prepare data for plotting
    if ious_with_crf is not None:
        # Include mean IoU
        ious_without_crf_with_mean = ious_without_crf + [np.mean(ious_without_crf)]
        ious_with_crf_with_mean = ious_with_crf + [np.mean(ious_with_crf)]

        x = np.arange(len(class_names))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, ious_without_crf_with_mean, width, label='Without CRF')
        plt.bar(x + width / 2, ious_with_crf_with_mean, width, label='With CRF')

        plt.ylabel('IoU Score')
        plt.title('IoU Comparison: With vs Without CRF')
        plt.xticks(x, class_names)
        plt.legend()

        # Add values on top of bars
        for i, v in enumerate(ious_without_crf_with_mean):
            plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center')

        for i, v in enumerate(ious_with_crf_with_mean):
            plt.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center')

        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    else:
        # Just plot without CRF if CRF results not available
        ious_without_crf_with_mean = ious_without_crf + [np.mean(ious_without_crf)]

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(class_names)), ious_without_crf_with_mean)

        plt.ylabel('IoU Score')
        plt.title('IoU Scores Without CRF')
        plt.xticks(np.arange(len(class_names)), class_names)

        # Add values on top of bars
        for i, v in enumerate(ious_without_crf_with_mean):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

        plt.grid(True, linestyle='--', alpha=0.6, axis='y')

    plt.tight_layout()
    plt.savefig('iou_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main function to run the pipeline
def main():
    print("Starting multispectral weed segmentation with CRF enhancement")

    # Configure for better performance
    configure_for_performance()

    # Set parameters
    RGB_DIR = "/Users/varad/Documents/RESEARCH-PROJECT/jesi_05_18/rgb"
    NIR_DIR = "/Users/varad/Documents/RESEARCH-PROJECT/jesi_05_18/nir"
    MASK_DIR = "/Users/varad/Documents/RESEARCH-PROJECT/jesi_05_18/gt"
    IMG_SIZE = (128, 128)  
    MAX_IMAGES = None 
    BATCH_SIZE = 4  
    EPOCHS = 70 
    AUGMENTATION_FACTOR = 3  

    try:
        # Load dataset
        print(f"Loading dataset from: {RGB_DIR}, {NIR_DIR}, {MASK_DIR}")
        X, Y = load_and_preprocess_multispectral_images(
            RGB_DIR, NIR_DIR, MASK_DIR, img_size=IMG_SIZE, max_images=MAX_IMAGES
        )

        print(f"Dataset loaded: {X.shape[0]} images of size {X.shape[1]}x{X.shape[2]} with {X.shape[3]} channels")

        # Split dataset into train, validation, and test sets
        # First split into train and temp (80% train, 20% temp)
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        # Then split temp into validation and test (50% validation, 50% test, which is 10% each of the original dataset)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=42
        )

        print(f"Dataset split: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test")

        # Apply data augmentation to training set
        print("Applying data augmentation...")
        X_train_aug, Y_train_aug = augment_data(
            X_train, Y_train, augmentation_factor=AUGMENTATION_FACTOR
        )

        print(f"Training set after augmentation: {X_train_aug.shape[0]} images")

        # Calculate class weights for handling class imbalance
        class_pixels = np.sum(Y_train_aug, axis=(0, 1, 2))
        total_pixels = np.sum(class_pixels)
        # Use inverse frequency with additional weighting for the minority class (weed)
        weights = np.array([
            1.0,  
            total_pixels / (class_pixels[1] * 3),  
            total_pixels / (class_pixels[2] * 3) * 2  
        ])
        # Normalize weights
        weights = weights / np.sum(weights) * 3

        print(f"Class distribution - Soil: {class_pixels[0]}, Crop: {class_pixels[1]}, Weed: {class_pixels[2]}")
        print(f"Class weights: {weights}")

        # Train model
        model, history = train_unet_model(
            X_train_aug, Y_train_aug, X_val, Y_val,
            img_size=IMG_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS,
            class_weights=weights
        )

        # Plot training history
        plot_training_history(history)

        # Evaluate model on test set
        Y_pred, refined_preds, ious_without_crf, ious_with_crf = evaluate_model(
            model, X_test, Y_test, apply_crf_postprocessing=USE_CRF
        )

        # Plot IoU comparison
        plot_iou_comparison(ious_without_crf, ious_with_crf)

        # Visualize results
        visualize_results(X_test, Y_test, Y_pred, refined_preds, num_samples=5)

        # Save model
        model.save('final_unet_crf_model.h5')
        print("Model saved as 'final_unet_crf_model.h5'")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

        # If the error is related to loading the dataset, create a synthetic dataset for testing
        print("Creating synthetic dataset for testing...")

        # Create synthetic dataset
        num_samples = 20
        X = np.random.rand(num_samples, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
        Y = np.zeros((num_samples, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

        # Create random masks with imbalanced classes
        for i in range(num_samples):
            # Random mask with more soil than crop than weed
            mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
            # Soil (majority class)
            mask[:, :] = a = np.random.rand(IMG_SIZE[0], IMG_SIZE[1]) < 0.7
            # Crop (medium class)
            mask[np.random.rand(IMG_SIZE[0], IMG_SIZE[1]) > 0.7] = 1
            # Weed (minority class - less than 10%)
            mask[np.random.rand(IMG_SIZE[0], IMG_SIZE[1]) > 0.9] = 2

            # Convert to one-hot encoding
            for j in range(3):
                Y[i, :, :, j] = (mask == j).astype(np.float32)

        # Split dataset
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

        print(
            f"Synthetic dataset created and split: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test")

        # Apply augmentation
        X_train_aug, Y_train_aug = augment_data(X_train, Y_train, augmentation_factor=2)

        # Calculate class weights
        class_pixels = np.sum(Y_train_aug, axis=(0, 1, 2))
        weights = np.array([1.0, 2.0, 4.0])  # Manual weights for synthetic data
        weights = weights / np.sum(weights) * 3

        print(f"Running with synthetic data and weights: {weights}")

        # Train model with fewer epochs for testing
        model, history = train_unet_model(
            X_train_aug, Y_train_aug, X_val, Y_val,
            img_size=IMG_SIZE, batch_size=BATCH_SIZE, epochs=10,
            class_weights=weights
        )

        # Evaluate briefly
        Y_pred, refined_preds, ious_without_crf, ious_with_crf = evaluate_model(
            model, X_test, Y_test, apply_crf_postprocessing=USE_CRF
        )

        # Visualize results
        visualize_results(X_test, Y_test, Y_pred, refined_preds, num_samples=3)


# Run the script if executed directly
if __name__ == "__main__":
    main()
