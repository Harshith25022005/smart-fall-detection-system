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
STRIDE       = 25
FALL_STRIDE  = 1        # dense windowing on fall data
NUM_FEATURES = 4        # x, y, z, magnitude

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
#
# KEY FIX vs previous version:
#
#   OLD: fall windows were mostly post-fall quiet data.
#        Post-fall (lying still) looks identical to
#        Sitting in WISDM — same mag, same variance.
#        Model learned "quiet ≈ Sitting" from 60k
#        sitting samples, overriding 585 fall windows.
#
#   NEW: The new synthetic_falling_v2.csv has 4106
#        samples with impact placed mid-sequence.
#        With stride=1 this gives ~2200 windows of
#        which 72% CONTAIN the impact spike + the
#        pre-fall or post-fall transition.
#        The model now learns: fall = a specific
#        TRAJECTORY through the window, not just
#        a quiet end-state.
#
#   ALSO: class_weight for Falling raised from 8→15
#         because Sitting has 2396 windows and we
#         have ~2200 fall windows — still 1:1 but
#         the quality of fall windows is now much
#         higher so a higher weight helps the model
#         take falling seriously vs Sitting confusion.
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

        print(f"\n  {activity}")
        print(f"    Raw samples : {len(arr)}")
        print(f"    Stride      : {stride}")

        count = 0

        for start in range(0, len(arr) - WINDOW_SIZE, stride):

            window = arr[start:start + WINDOW_SIZE]

            if len(window) == WINDOW_SIZE:
                X.append(window)
                y.append(activity)
                count += 1

        print(f"    Windows     : {count}")

    return np.array(X), np.array(y)


# =====================================================
# MODEL
# =====================================================

def build_model():

    model = tf.keras.Sequential([

        # Block 1 — local feature extraction
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

    # ------------------------------------------------
    # PATHS — update if needed
    # ------------------------------------------------

    wisdm_path = "../../datasets/wisdm_dataset/WISDM_ar_v1.1_raw.txt"
    fall_path  = "../../datasets/synthetic_fall_dataset/synthetic_falling_v2.csv"
    # ------------------------------------------------
    # LOAD
    # ------------------------------------------------

    print("\n================================")
    print("LOADING WISDM")
    print("================================")

    wisdm_data = load_raw(wisdm_path)
    print(f"  WISDM samples : {len(wisdm_data)}")

    print("\n================================")
    print("LOADING FALL DATA")
    print("================================")

    fall_data = load_raw(fall_path)
    print(f"  Fall samples  : {len(fall_data)}")

    data = wisdm_data + fall_data
    print(f"  Total samples : {len(data)}")

    # ------------------------------------------------
    # WINDOWING
    # ------------------------------------------------

    print("\n================================")
    print("CREATING WINDOWS")
    print("================================")

    X, y = create_windows(data)

    print(f"\n  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # ------------------------------------------------
    # NORMALIZE
    # ------------------------------------------------

    print("\n================================")
    print("NORMALIZING")
    print("================================")

    X_flat = X.reshape(-1, NUM_FEATURES)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X      = X_flat.reshape(-1, WINDOW_SIZE, NUM_FEATURES)

    print("\n  🔥 COPY THESE INTO YOUR ESP32 CODE:")
    print(f"  float mean[4] = {{{', '.join(f'{v:.8f}' for v in scaler.mean_)}}};")
    print(f"  float stdv[4] = {{{', '.join(f'{v:.8f}' for v in scaler.scale_)}}};")

    # ------------------------------------------------
    # ENCODE LABELS
    # ------------------------------------------------

    le = LabelEncoder()
    y  = le.fit_transform(y)

    print(f"\n  Classes : {list(le.classes_)}")
    print(f"  Mapping : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ------------------------------------------------
    # SHUFFLE + SPLIT
    # ------------------------------------------------

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\n  Train: {X_train.shape}  Test: {X_test.shape}")

    # ------------------------------------------------
    # CLASS WEIGHTS
    #
    # Falling has ~2200 windows, Sitting has ~2396.
    # The danger is still that Sitting dominates because
    # WISDM has 60k sitting SAMPLES creating very rich
    # sitting features. Weight=15 brings effective
    # contribution in line with the larger classes.
    # ------------------------------------------------

    class_weights = {}
    for i, cls in enumerate(le.classes_):
        class_weights[i] = 15 if cls == "Falling" else 1

    print(f"\n  Class weights: {class_weights}")

    # ------------------------------------------------
    # TRAIN
    # ------------------------------------------------

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

    # ------------------------------------------------
    # EVALUATE
    # ------------------------------------------------

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n  Final accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n================================")
    print("CLASSIFICATION REPORT")
    print("================================")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\n================================")
    print("CONFUSION MATRIX")
    print("================================")

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Check specifically: how many Falling windows
    # get misclassified as Sitting?
    fall_idx = list(le.classes_).index("Falling")
    sit_idx  = list(le.classes_).index("Sitting")
    fall_as_sit = cm[fall_idx][sit_idx]
    total_fall  = cm[fall_idx].sum()

    print(f"\n  Falling misclassified as Sitting: {fall_as_sit}/{total_fall}")
    print(f"  ({100*fall_as_sit/total_fall:.1f}%  — target is <5%)")

    # ------------------------------------------------
    # SAVE
    # ------------------------------------------------

    model.save("model_fall_detection.h5")
    print("\n  Saved: model_fall_detection.h5")
    print("  DONE")


if __name__ == "__main__":
    main()