import torch

from models.MyNet.net_desc import create_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(mode="MyNet", nr_types=3).to(device)
    model.eval()

    x = torch.zeros(1, 3, 256, 256, device=device)
    with torch.no_grad():
        outputs = model(x)

    print("Device:", device)
    for name, value in outputs.items():
        print(f"{name}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()

