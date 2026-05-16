import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# =====================================================
# CONFIG
# =====================================================

WINDOW_SIZE  = 50
STRIDE       = 25       # stride for normal activities

# 🔥 FALL USES A TIGHTER STRIDE
# With 635 clean samples:
#   stride=25 →  23 windows  (broken — 318:1 imbalance)
#   stride=5  → 117 windows  (much better — 62:1 imbalance)
#   stride=1  → 585 windows  (best — 12:1 imbalance, handled by class weight)
FALL_STRIDE  = 1

NUM_FEATURES = 4   # x, y, z, magnitude

CLASSES = [
    "Walking",
    "Jogging",
    "Sitting",
    "Standing",
    "Falling"
]

# =====================================================
# PARSE DATA
# =====================================================

def parse_line(line):

    line = line.strip().replace(";", "")

    if not line:
        return None

    parts = line.split(",")

    if len(parts) != 6:
        return None

    try:
        activity = parts[1].strip()
        x        = float(parts[3])
        y        = float(parts[4])
        z        = float(parts[5])
        mag      = np.sqrt(x*x + y*y + z*z)

        return activity, x, y, z, mag

    except:
        return None


# =====================================================
# LOAD FILE
# =====================================================

def load_raw(path):

    data = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = parse_line(line)
            if rec:
                data.append(rec)

    return data


# =====================================================
# CREATE WINDOWS
# Per-class stride: Falling uses FALL_STRIDE=1
# everything else uses STRIDE=25
# =====================================================

def create_windows(data):

    X = []
    y = []

    grouped = {}

    for activity, x, y_val, z, mag in data:

        if activity not in CLASSES:
            continue

        if activity not in grouped:
            grouped[activity] = []

        grouped[activity].append([x, y_val, z, mag])

    for activity in grouped:

        arr    = np.array(grouped[activity])
        stride = FALL_STRIDE if activity == "Falling" else STRIDE

        print(f"\n{activity}")
        print(f"  Raw samples : {len(arr)}")
        print(f"  Stride used : {stride}")

        count = 0

        for start in range(0, len(arr) - WINDOW_SIZE, stride):

            end    = start + WINDOW_SIZE
            window = arr[start:end]

            if len(window) == WINDOW_SIZE:
                X.append(window)
                y.append(activity)
                count += 1

        print(f"  Windows     : {count}")

    return np.array(X), np.array(y)


# =====================================================
# MODEL
# =====================================================

def build_model():

    model = tf.keras.Sequential([

        # Block 1
        tf.keras.layers.Conv1D(
            32, 5,
            activation='relu',
            input_shape=(WINDOW_SIZE, NUM_FEATURES)
        ),
        tf.keras.layers.BatchNormalization(),

        # Block 2
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),

        # Block 3
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),

        # Dense
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =====================================================
# MAIN
# =====================================================

def main():

    wisdm_path = "../../datasets/wisdm_dataset/WISDM_ar_v1.1_raw.txt"
    fall_path  = "../../datasets/synthetic_fall_dataset/synthetic_falling_clean.txt"
    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------

    print("\n================================")
    print("LOADING WISDM")
    print("================================")

    wisdm_data = load_raw(wisdm_path)
    print(f"WISDM samples: {len(wisdm_data)}")

    print("\n================================")
    print("LOADING FALL DATA")
    print("================================")

    fall_data = load_raw(fall_path)
    print(f"Fall samples : {len(fall_data)}")

    data = wisdm_data + fall_data
    print(f"Total samples: {len(data)}")

    # --------------------------------------------------
    # WINDOWING
    # --------------------------------------------------

    print("\n================================")
    print("CREATING WINDOWS")
    print("================================")

    X, y = create_windows(data)

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # --------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------

    print("\n================================")
    print("NORMALIZING")
    print("================================")

    X = X.reshape(-1, NUM_FEATURES)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(-1, WINDOW_SIZE, NUM_FEATURES)

    print("\n🔥 COPY THESE INTO YOUR ESP32 CODE:")
    print(f"float mean[4] = {{{', '.join(f'{v:.8f}' for v in scaler.mean_)}}};")
    print(f"float stdv[4] = {{{', '.join(f'{v:.8f}' for v in scaler.scale_)}}};")

    # --------------------------------------------------
    # ENCODE LABELS
    # --------------------------------------------------

    le = LabelEncoder()
    y  = le.fit_transform(y)

    print(f"\nClasses: {list(le.classes_)}")
    print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # --------------------------------------------------
    # SHUFFLE + SPLIT
    # --------------------------------------------------

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

    # --------------------------------------------------
    # CLASS WEIGHTS
    # With stride=1 we get ~585 fall windows vs ~7300 per other class
    # Ratio is ~12:1 so weight=8 brings effective ratio to ~1.5:1
    # --------------------------------------------------

    class_weights = {}
    for i, cls in enumerate(le.classes_):
        class_weights[i] = 8 if cls == "Falling" else 1

    print(f"\nClass weights: {class_weights}")

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=callbacks
    )

    # --------------------------------------------------
    # EVALUATE
    # --------------------------------------------------

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nFinal accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n================================")
    print("CLASSIFICATION REPORT")
    print("================================")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\n================================")
    print("CONFUSION MATRIX")
    print("================================")
    print(confusion_matrix(y_test, y_pred))

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    model.save("model_fall_detection.h5")
    print("\nSaved: model_fall_detection.h5")
    print("DONE")


if __name__ == "__main__":
    main()