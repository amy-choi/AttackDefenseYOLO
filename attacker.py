"""
@InProceedings{choi2022advYOLO,
  title = {Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios},
  author = {Choi, Jung Im and Tian, Qing},
  booktitle = {2022 IEEE Intelligent Vehicles Symposium (IV)},
  year = {2022},
  pages = {1011-1017},
  doi={10.1109/IV51971.2022.9827222},
}
"""


import torch

IMAGE_SCALE = 2.0 / 255


class FGSMAttacker:
    def __init__(self, epsilon=1, device="cuda:0", attack_type="total"):
        self.epsilon = epsilon * IMAGE_SCALE
        self.device = device
        self.attack_type = attack_type

    def attack(self, image_clean, label, yolo_loss, model):
        image_clean = image_clean.to(self.device)  # .cuda()
        delta = (
            torch.zeros_like(image_clean)
            .uniform_(-self.epsilon, self.epsilon)
            .to(self.device)
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)
        delta.requires_grad = True

        outputs = model(image_clean + delta)
        losses_cls = 0
        losses_loc = 0
        losses_conf = 0
        losses_total = 0
        num_pos_all = 0

        for l in range(len(outputs)):
            loss_total, loss_loc, loss_conf, loss_cls, num_pos = yolo_loss(
                l, outputs[l], label
            )
            losses_cls += loss_cls
            losses_loc += loss_loc
            losses_conf += loss_conf
            losses_total += loss_total
            num_pos_all += num_pos

        if self.attack_type == "cls":
            loss = losses_cls / num_pos_all
        elif self.attack_type == "loc":
            loss = losses_loc / num_pos_all
        elif self.attack_type == "conf":
            loss = losses_conf / num_pos_all
        elif self.attack_type == "total":
            loss = losses_total / num_pos_all
        elif self.attack_type == "conf_loc":
            loss = (losses_conf + losses_loc) / num_pos_all
        elif self.attack_type == "conf_cls":
            loss = (losses_conf + losses_cls) / num_pos_all
        elif self.attack_type == "cls_loc":
            loss = (losses_cls + losses_loc) / num_pos_all

        loss.backward()
        grad = delta.grad.detach()

        delta.data = torch.clamp(
            delta + self.epsilon * torch.sign(grad), -self.epsilon, self.epsilon
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)
        adv_img = (image_clean + delta).detach()

        return adv_img

    def generate_cls(self, image_clean, label, yolo_loss, model):
        image_clean = image_clean.clone().detach().to(self.device)  # .cuda()
        delta = (
            torch.zeros_like(image_clean)
            .uniform_(-self.epsilon, self.epsilon)
            .to(self.device)
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)
        delta.requires_grad = True

        outputs = model(image_clean + delta)
        losses = 0
        num_pos_all = 0

        for l in range(len(outputs)):
            _, _, _, loss_cls, num_pos = yolo_loss(l, outputs[l], label)
            losses += loss_cls
            num_pos_all += num_pos

        loss = losses / num_pos_all

        loss.backward()
        grad = delta.grad.detach()

        delta.data = torch.clamp(
            delta + self.epsilon * torch.sign(grad), -self.epsilon, self.epsilon
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)
        adv_img = (image_clean + delta).detach()

        return adv_img

    def generate_conf(self, image_clean, label, yolo_loss, model):
        image_clean = image_clean.clone().detach().to(self.device)  # .cuda()
        delta = (
            torch.zeros_like(image_clean)
            .uniform_(-self.epsilon, self.epsilon)
            .to(self.device)
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)
        delta.requires_grad = True

        outputs = model(image_clean + delta)
        losses = 0
        num_pos_all = 0

        for l in range(len(outputs)):
            _, _, loss_conf, _, num_pos = yolo_loss(l, outputs[l], label)
            losses += loss_conf
            num_pos_all += num_pos

        loss = losses / num_pos_all

        loss.backward()
        grad = delta.grad.detach()

        delta.data = torch.clamp(
            delta + self.epsilon * torch.sign(grad), -self.epsilon, self.epsilon
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)
        adv_img = (image_clean + delta).detach()

        return adv_img

    def generate_loc(self, image_clean, label, yolo_loss, model):
        image_clean = image_clean.clone().detach().to(self.device)  # .cuda()
        delta = (
            torch.zeros_like(image_clean).uniform_(-self.epsilon, self.epsilon).cuda()
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)
        delta.requires_grad = True

        outputs = model(image_clean + delta)
        losses = 0
        num_pos_all = 0

        for l in range(len(outputs)):
            _, loss_loc, _, _, num_pos = yolo_loss(l, outputs[l], label)
            losses += loss_loc
            num_pos_all += num_pos

        loss = losses / num_pos_all

        loss.backward()
        grad = delta.grad.detach()

        delta.data = torch.clamp(
            delta + self.epsilon * torch.sign(grad), -self.epsilon, self.epsilon
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)
        adv_img = (image_clean + delta).detach()

        return adv_img


class PGDAttacker:
    def __init__(
        self,
        num_iter=1,
        epsilon=1,
        step_size=1,
        prob_start_from_clean=0.0,
        device="cuda:0",
        attack_type="total",
    ):
        step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size * IMAGE_SCALE
        self.device = device
        self.attack_type = attack_type

    def attack(self, image_clean, label, yolo_loss, model):
        image_clean = image_clean.clone().detach().to(self.device)
        delta = (
            torch.zeros_like(image_clean)
            .uniform_(-self.epsilon, self.epsilon)
            .to(self.device)
        )
        delta.data = torch.max(torch.min(1 - image_clean, delta.data), 0 - image_clean)

        delta.requires_grad = True
        for _ in range(self.num_iter):
            outputs = model(image_clean + delta)

            losses_cls = 0
            losses_loc = 0
            losses_conf = 0
            losses_total = 0
            num_pos_all = 0

            for l in range(len(outputs)):
                loss_total, loss_loc, loss_conf, loss_cls, num_pos = yolo_loss(
                    l, outputs[l], label
                )
                losses_cls += loss_cls
                losses_loc += loss_loc
                losses_conf += loss_conf
                losses_total += loss_total
                num_pos_all += num_pos

            if self.attack_type == "cls":
                loss = losses_cls / num_pos_all
            elif self.attack_type == "loc":
                loss = losses_loc / num_pos_all
            elif self.attack_type == "conf":
                loss = losses_conf / num_pos_all
            elif self.attack_type == "total":
                loss = losses_total / num_pos_all
            elif self.attack_type == "conf_loc":
                loss = (losses_conf + losses_loc) / num_pos_all
            elif self.attack_type == "conf_cls":
                loss = (losses_conf + losses_cls) / num_pos_all
            elif self.attack_type == "cls_loc":
                loss = (losses_cls + losses_loc) / num_pos_all

            loss.backward()
            grad = delta.grad.detach()
            delta.data = torch.clamp(
                delta + self.step_size * torch.sign(grad), -self.epsilon, self.epsilon
            )
            delta.data = torch.max(
                torch.min(1 - image_clean, delta.data), 0 - image_clean
            )
            delta.grad.zero_()
        adv_img = (image_clean + delta).detach()

        return adv_img
