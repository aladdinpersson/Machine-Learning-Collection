import torch
import visdom
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32


def save_checkpoint(filename, model, optimizer, train_acc, epoch):
    save_state = {
        "state_dict": model.state_dict(),
        "acc": train_acc,
        "epoch": epoch + 1,
        "optimizer": optimizer.state_dict(),
    }
    print()
    print("Saving current parameters")
    print("___________________________________________________________")

    torch.save(save_state, filename)


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training or validation set")
    else:
        print("Checking accuracy on test set")
    num_correct = 0
    num_samples = 0
    # model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = (float(num_correct) / num_samples) * 100.0
        print("Got %d / %d correct (%.2f)" % (num_correct, num_samples, acc))
        return acc


def load_model(args, model, optimizer):
    if args.resume:
        model.eval()
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            best_acc = checkpoint["acc"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            return model, optimizer, checkpoint, start_epoch, best_acc
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("No pretrained model. Starting from scratch!")


class visdom_plotting(object):
    def __init__(self):
        self.viz = visdom.Visdom()

        self.cur_batch_win = None
        self.cur_batch_win_opts = {
            "title": "Epoch Loss Trace",
            "xlabel": "Batch Number",
            "ylabel": "Loss",
            "width": 600,
            "height": 400,
        }

        self.cur_validation_acc = None
        self.cur_validation_acc_opts = {
            "title": "Validation accuracy",
            "xlabel": "Epochs",
            "ylabel": "Validation Accuracy",
            "width": 600,
            "height": 400,
        }

        self.cur_training_acc = None
        self.cur_training_acc_opts = {
            "title": "Training accuracy",
            "xlabel": "Epochs",
            "ylabel": "Train Accuracy",
            "width": 600,
            "height": 400,
        }

    def create_plot(
        self, loss_list, batch_list, validation_acc_list, epoch_list, training_acc_list
    ):

        if self.viz.check_connection():
            self.cur_batch_win = self.viz.line(
                torch.FloatTensor(loss_list),
                torch.FloatTensor(batch_list),
                win=self.cur_batch_win,
                name="current_batch_loss",
                update=(None if self.cur_batch_win is None else "replace"),
                opts=self.cur_batch_win_opts,
            )

            self.cur_validation_acc = self.viz.line(
                torch.FloatTensor(validation_acc_list),
                torch.FloatTensor(epoch_list),
                win=self.cur_validation_acc,
                name="current_validation_accuracy",
                update=(None if self.cur_validation_acc is None else "replace"),
                opts=self.cur_validation_acc_opts,
            )

            self.cur_training_acc = self.viz.line(
                torch.FloatTensor(training_acc_list),
                torch.FloatTensor(epoch_list),
                win=self.cur_validation_acc,
                name="current_training_accuracy",
                update=(None if self.cur_training_acc is None else "replace"),
                opts=self.cur_training_acc_opts,
            )


#
