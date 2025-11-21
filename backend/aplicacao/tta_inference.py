import torch
import torch.nn.functional as F
from torchvision import transforms

class TTAPredictor:
    """Test-Time Augmentation para aumentar acurácia em ~2-3%."""
    def __init__(self, model, num_augmentations: int = 5, img_size: int = 224):
        self.model = model
        self.num_augmentations = max(1, num_augmentations)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.base_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_list = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]

    @torch.no_grad()
    def predict_logits(self, pil_image):
        """Retorna logits médios das variações TTA."""
        logits_list = []
        # Passagem base
        x0 = self.base_tf(pil_image).unsqueeze(0).to(self.device)
        logits_list.append(self.model(x0))
        # Augmentações
        for i in range(self.num_augmentations):
            aug_tf = transforms.Compose([
                transforms.Resize((self.base_tf.transforms[0].size[0], self.base_tf.transforms[0].size[1])),
                self.aug_list[i % len(self.aug_list)],
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            xi = aug_tf(pil_image).unsqueeze(0).to(self.device)
            logits_list.append(self.model(xi))
        return torch.stack(logits_list, dim=0).mean(dim=0)

    @torch.no_grad()
    def predict_proba(self, pil_image):
        logits = self.predict_logits(pil_image)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_class(self, pil_image):
        probs = self.predict_proba(pil_image)
        conf, pred = probs.max(dim=1)
        return pred.item(), conf.item(), probs.squeeze(0).tolist()
