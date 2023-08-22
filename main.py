from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
import torch
import matplotlib.pyplot as plt
from config import (num_epochs, device,
                    model, train_loader, val_loader,
                    loss_fn, optimizer, l1, dice, edge,
                    pet_metric, ct_metric,
                    save_dir)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def train():
    model.train()
    epoch_pet_l1_loss, epoch_pet_ce_loss, epoch_pet_dice_loss, epoch_pet_edge_loss = 0, 0, 0, 0
    epoch_ct_l1_loss, epoch_ct_ce_loss, epoch_ct_dice_loss, epoch_ct_edge_loss = 0, 0, 0, 0
    step = 0
    for batch_data in train_loader:
        optimizer.zero_grad()
        pet_images, pet_labels, ct_images, ct_labels = (
            batch_data["pet_image"].to(device),
            batch_data["pet_label"].to(device),
            batch_data["ct_image"].to(device),
            batch_data["ct_label"].to(device),
        )
        pet_outputs, ct_outputs = model(pet_images, ct_images)
        pet_l1_loss, pet_ce_loss, pet_dice_loss, pet_edge_loss = loss_fn(
            pet_outputs, pet_labels)
        ct_l1_loss, ct_ce_loss, ct_dice_loss, ct_edge_loss = loss_fn(
            ct_outputs, ct_labels)
        loss = l1*pet_l1_loss+pet_ce_loss+dice*pet_dice_loss+edge*pet_edge_loss + \
            l1*ct_l1_loss+ct_ce_loss+dice*ct_dice_loss+edge*ct_edge_loss
        loss.backward()
        optimizer.step()
        epoch_pet_l1_loss += pet_l1_loss.item()
        epoch_pet_ce_loss += pet_ce_loss.item()
        epoch_pet_dice_loss += pet_dice_loss.item()
        epoch_pet_edge_loss += pet_edge_loss.item()
        epoch_ct_l1_loss += ct_l1_loss.item()
        epoch_ct_ce_loss += ct_ce_loss.item()
        epoch_ct_dice_loss += ct_dice_loss.item()
        epoch_ct_edge_loss += ct_edge_loss.item()
        step += 1
    epoch_pet_l1_loss /= step
    epoch_pet_ce_loss /= step
    epoch_pet_dice_loss /= step
    epoch_pet_edge_loss /= step
    epoch_ct_l1_loss /= step
    epoch_ct_ce_loss /= step
    epoch_ct_dice_loss /= step
    epoch_ct_edge_loss /= step
    return [epoch_pet_l1_loss, epoch_pet_ce_loss, epoch_pet_dice_loss, epoch_pet_edge_loss], [epoch_ct_l1_loss, epoch_ct_ce_loss, epoch_ct_dice_loss, epoch_ct_edge_loss]


post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])


def evaluate():
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            pet_images, pet_labels, ct_images, ct_labels = (
                batch_data["pet_image"].to(device),
                batch_data["pet_label"].to(device),
                batch_data["ct_image"].to(device),
                batch_data["ct_label"].to(device),
            )

            pet_outputs = torch.tensor([]).to(device)
            ct_outputs = torch.tensor([]).to(device)
            for depth in range(pet_images.shape[-1]):
                pet_image = pet_images[:, :, :, :, depth]
                ct_image = ct_images[:, :, :, :, depth]
                pet_output, ct_output = model(pet_image, ct_image)
                pet_outputs = torch.concat(
                    [pet_outputs, torch.unsqueeze(pet_output, dim=-1)], dim=-1)
                ct_outputs = torch.concat(
                    [ct_outputs, torch.unsqueeze(ct_output, dim=-1)], dim=-1)

            pet_outputs = [post_pred(i) for i in decollate_batch(pet_outputs)]
            ct_outputs = [post_pred(i) for i in decollate_batch(ct_outputs)]
            pet_labels = [post_label(i) for i in decollate_batch(pet_labels)]
            ct_labels = [post_label(i) for i in decollate_batch(ct_labels)]

            pet_metric(y_pred=pet_outputs, y=pet_labels)
            ct_metric(y_pred=ct_outputs, y=ct_labels)

        pet_mean_metric = pet_metric.aggregate().item()
        ct_mean_metric = ct_metric.aggregate().item()
        pet_metric.reset()
        ct_metric.reset()
        return pet_mean_metric, ct_mean_metric


best_metric = {'pet': -1, 'ct': -1}
best_epoch = {'pet': -1, 'ct': -1}
pet_loss_values, ct_loss_values, pet_metric_values, ct_metric_values = [], [], [], []


def update(epoch, pet_loss, ct_loss, pet_metric, ct_metric):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{num_epochs}")
    print(
        f"pet:l1_loss={pet_loss[0]:.4f},ce_loss={pet_loss[1]:.4f},dice_loss={pet_loss[2]:.4f},edge_loss={pet_loss[3]:.4f},\nct:l1_loss={ct_loss[0]:.4f},ce_loss={ct_loss[1]:.4f},dice_loss={ct_loss[2]:.4f},edge_loss={ct_loss[3]:.4f}")
    pet_loss_values.append(pet_loss)
    ct_loss_values.append(ct_loss)
    pet_metric_values.append(pet_metric)
    ct_metric_values.append(ct_metric)
    print(f"pet_val_dice:{pet_metric:.4f},ct_val_dice:{ct_metric:.4f}")
    with open(save_dir+"/log.txt", "a") as f:
        f.write(f"pet_val_dice:{pet_metric:.4f},ct_val_dice:{ct_metric:.4f}\n")
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train Average Loss")
    plt.xlabel("epoch")
    color = ["red", "yellow", "blue", "green"]
    pet_loss = ["pet_l1", "pet_ce", "pet_dice", "pet_edge"]
    ct_loss = ["ct_l1", "ct_ce", "ct_dice", "ct_edge"]
    for j in range(len(pet_loss)):
        x = [i + 1 for i in range(len(pet_loss_values))]
        y = [pet_loss_values[i][j] for i in range(len(pet_loss_values))]
        z = [ct_loss_values[i][j] for i in range(len(ct_loss_values))]
        plt.plot(x, y, color=color[j], linestyle="-", label=pet_loss[j])
        plt.plot(x, z, color=color[j], linestyle=":", label=ct_loss[j])
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [(i + 1) for i in range(len(pet_metric_values))]
    plt.xlabel("epoch")
    plt.plot(x, pet_metric_values)
    plt.plot(x, ct_metric_values)
    plt.savefig(save_dir+"/curve.png")
    if pet_metric > best_metric['pet']:
        best_metric['pet'] = pet_metric
        best_epoch["pet"] = epoch + 1
        if pet_metric > 0.7:
            torch.save(model.state_dict(), save_dir +
                       "/pet_best_model_params.pth")
        print("saved new pet best metric model")
    if ct_metric > best_metric['ct']:
        best_metric['ct'] = ct_metric
        best_epoch["ct"] = epoch + 1
        if ct_metric > 0.7:
            torch.save(model.state_dict(), save_dir +
                       "/ct_best_model_params.pth")
        print("saved new ct best metric model")


def record():
    with open(save_dir+"/log.txt", "a") as f:
        f.write(
            f"best_pet_metric{best_metric['pet']:.4f},at epoch{best_epoch['pet']}\nbest_ct_metric{best_metric['ct']:.4f},at epoch{best_epoch['ct']}\n")
    print("best_pet_metric", best_metric["pet"], "at epoch", best_epoch["pet"])
    print("best_ct_metric", best_metric["ct"], "at epoch", best_epoch["ct"])


for epoch in range(num_epochs):
    pet_loss_value, ct_loss_value = train()
    pet_metric_value, ct_metric_value = evaluate()
    update(epoch, pet_loss_value, ct_loss_value,
           pet_metric_value, ct_metric_value)
record()
