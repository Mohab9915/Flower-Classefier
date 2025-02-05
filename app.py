
import tensorflow as tf
from predict import predict
from load_class import load_class_names
import argparse
from tf_keras.models import load_model
import tensorflow_hub as hub

def main():
    parser = argparse.ArgumentParser(description="Flower Classifier Prediction")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("model_path", help="Path to the trained model (.h5 file)")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to a JSON file mapping labels to class names")

    args = parser.parse_args()

    try:
        custom_objects = {
            'KerasLayer': hub.KerasLayer,
            'keras_layer': hub.KerasLayer
        }
        
        model = load_model(
            args.model_path,
            custom_objects=custom_objects,
            compile=False
        )
        probs, classes = predict(args.image_path, model, args.top_k)

        if args.category_names:
            class_names = load_class_names(args.category_names)
            class_labels = [class_names[str(cls)] for cls in classes]
        else:
            class_labels = classes

        print("\nTop Predictions:")
        for i in range(args.top_k):
            print(f"{class_labels[i]}: {probs[i]:.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()