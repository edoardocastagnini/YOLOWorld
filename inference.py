import argparse
from ultralytics import YOLOWorld
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="YOLOWorld Inference Script")
    parser.add_argument('--model', type=str, required=True, help='Path to custom .pt model file')
    parser.add_argument('--image', type=str, required=True, help='Path to input .jpg image')
    parser.add_argument('--output', type=str, default='result.jpg', help='Path to save output image')
    parser.add_argument('--classes', type=str, help='Comma-separated list of classes for open vocabulary')
    args = parser.parse_args()

    # Load model
    model = YOLOWorld(args.model)

    # Set custom classes if provided
    if args.classes:
        class_list = [c.strip() for c in args.classes.split(',')]
        model.set_classes(class_list)

    # Load image
    img = Image.open(args.image).convert('RGB')

    # Run inference
    results = model(img)


    results[0].save(filename=args.output)

    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
