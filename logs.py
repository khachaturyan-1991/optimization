"""TensorBoard logging helpers."""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont


class Logs:

    def __init__(self, cfg) -> None:
        """Initialize writer and class labels."""
        log_dir = cfg["log_dir"]
        classes = cfg.get("classes")
        if not classes:
            classes = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
        self.classes = classes
        self.writer = SummaryWriter(log_dir)

    def log_loss(self, epoch, train_loss, test_loss, loss_mAP=None):
        """Log scalar losses (and optional loss_mAP)."""
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/test", test_loss, epoch)
        if loss_mAP is not None:
            self.writer.add_scalar("loss/loss_mAP", loss_mAP, epoch)

    def log_text(self, ckpt_path, epoch):
        """Log checkpoint path."""
        self.writer.add_text("checkpoint", ckpt_path, epoch)

    def log_predictions(self, images, labels, preds, epoch):
        """Log a grid of prediction images with titles."""
        if images is None or labels is None or preds is None:
            return
        
        # Take the first 5
        images = images[:5].cpu()
        labels = labels[:5].cpu().tolist()
        preds = preds[:5].cpu().tolist()

        labeled_images = []
        
        for i in range(len(images)):
            img_pil = F.to_pil_image(images[i])

            scale = 4 
            img_pil = img_pil.resize((img_pil.width * scale, img_pil.height * scale), resample=Image.NEAREST)

            text_space = 20
            width, height = img_pil.size
            canvas = Image.new("RGB", (width, height + text_space), color="white")
            canvas.paste(img_pil, (0, text_space))

            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()

            txt = f"T:{self.classes[labels[i]]}\nP:{self.classes[preds[i]]}"

            draw.text((2, 0), txt, fill="black", font=font)

            labeled_images.append(F.to_tensor(canvas))

        final_grid = torchvision.utils.make_grid(labeled_images, nrow=5, padding=2)
        self.writer.add_image("Predictions_Visual", final_grid, global_step=epoch)
