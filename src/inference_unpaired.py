import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform


def find_image_files(input_file_path):
    """
    Find all image files (jpg, png) in the given path.
    If input_file_path is a file, return it if it's an image.
    If input_file_path is a directory, recursively search for images.
    """
    input_file_path = Path(input_file_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    if input_file_path.is_file():
        if input_file_path.suffix in image_extensions:
            return [input_file_path]
        else:
            raise ValueError(f"Input file {input_file_path} is not a valid image file (jpg, jpeg, png)")

    elif input_file_path.is_dir():
        found_images = []
        # Folders to ignore
        ignore_folders = {'depth', 'label', 'labels'}

        # Search recursively
        for file_path in input_file_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in image_extensions:
                # Check if any parent folder should be ignored
                should_ignore = False
                for parent in file_path.parents:
                    if parent.name.lower() in ignore_folders:
                        should_ignore = True
                        break

                if not should_ignore:
                    found_images.append(file_path)
        return found_images

    else:
        raise ValueError(f"Input path {input_file_path} does not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to the input image file or directory')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if (args.model_name is None) == (args.model_path is None):
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()

    T_val = build_transform(args.image_prep)

    # Find all image files
    input_path = Path(args.input)
    output_path = Path(args.output_dir)
    image_files = find_image_files(input_path)

    if not image_files:
        print("No image files found!")
        exit(1)

    for idx, input_image_path in enumerate(tqdm(image_files, desc='Processing images')):
        input_image = Image.open(input_image_path).convert('RGB')

        # Calculate relative path to maintain folder structure
        if input_path.is_file():
            # If input was a single file, just use the filename
            relative_path = input_image_path.name
        else:
            # If input was a directory, preserve the relative structure
            relative_path = input_image_path.relative_to(input_path)

        # Create output path with preserved structure
        output_file_path = output_path / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if the image has been already processed
        if output_file_path.is_file():
            continue

        # translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            if args.use_fp16:
                x_t = x_t.half()
            output = model(x_t, direction=args.direction, caption=args.prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        # output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

        # Save the output image
        output_pil.save(output_file_path)

        # Each 10 images clear the CUDA cache to free memory
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
