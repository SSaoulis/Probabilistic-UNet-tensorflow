from load_dataset import train_val_test_split
from prob_unet import ProbUNet
import tensorflow as tf


def main():

    train_ds, val_ds, test_ds = train_val_test_split(
        "LIDC-IDRI-slices", total_dataset_size=40
    )
    print(len(train_ds), len(val_ds), len(test_ds))

    model = ProbUNet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("probunet_best", save_best_only=True),
        ],
    )

    test_metrics = model.evaluate(test_ds)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
