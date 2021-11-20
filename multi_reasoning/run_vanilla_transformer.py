import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from vanilla_transformer import ReconstructionLoss

from util import make_tgt_mask
from torch.utils.tensorboard import SummaryWriter


def calculate_acc(pred_y:torch.Tensor,true_y:torch.Tensor,pad_id=0)-> int:
    """
    Args:
        pred_y (torch.Tensor):output. shape of y is (batch_size, seq_len, dim)
        true_y (torch.Tensor):
        h (torch.Tensor) :
    Returns:
        int: num of accuracy
    """
  
    count = 0
    true_y= true_y[:,1:]
    pred_y = torch.argmax(pred_y[:,:-1], dim=-1)
        
    for pred_y_sequence,true_y_sequence in zip(pred_y,true_y):
        flag = False
        for pred_y_token,true_y_token in zip(pred_y_sequence,true_y_sequence):
            if true_y_token.item() != pad_id and pred_y_token.item() != true_y_token.item(): flag = True
        if flag == False : count += 1

    return count




def vanilla_train(
    model: nn.Module,
    optimizer: optim,
    train_data: Dataset,
    train_loader: DataLoader,
    device: torch.device,
    writer:SummaryWriter,
    valid: bool = False,
    valid_data: Dataset = None,
    valid_loader: DataLoader = None,
    epochs: int = 30,
    pad_id: int = 0,
    sep_id: int = None,
    modelsave_pass = "best_vanilla_models/",
):

    best_accuracy = 0.0
    loss_rec_inst = ReconstructionLoss()
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        total_acc = 0.0

        model.train()
        for x, true_y in train_loader:
            x = x.to(device)
            true_y = true_y.to(device)
            tgt_mask = make_tgt_mask(true_y.shape[1]).to(device)
            src_key_padding_mask = (x == pad_id)
            tgt_key_padding_mask = (true_y == pad_id)

            optimizer.zero_grad()
            pred_y = model(x, true_y, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            loss = loss_rec_inst(pred_y,true_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += calculate_acc(pred_y, true_y,pad_id=pad_id)
        
        writer.add_scalar("Loss/train", total_loss/len(train_loader), epoch)
        writer.add_scalar("Acc/train", total_acc/len(train_data), epoch)
        writer.flush()
        print(f"train_loss:{total_loss/len(train_loader)}")
        print(f"train_acc:{total_acc/len(train_data)}")

        if valid == True:
            with torch.no_grad():
                model.eval()
                valid_total_loss = 0.0
                valid_total_acc = 0.0
                for x, true_y in valid_loader:
                    x = x.to(device)
                    true_y = true_y.to(device)
                    tgt_mask = make_tgt_mask(true_y.shape[1]).to(device)
                    src_key_padding_mask = (x == pad_id)
                    tgt_key_padding_mask = (true_y == pad_id)

                    pred_y = model(
                        x, true_y, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                    
                    loss = loss_rec_inst(pred_y,true_y)
                    valid_total_loss += loss.item()
                    valid_total_acc += calculate_acc(pred_y, true_y,pad_id=pad_id)
                
                
                if best_accuracy < valid_total_acc/len(valid_data): 
                    best_accuracy = valid_total_acc/len(valid_data)
                    torch.save(model.state_dict(), modelsave_pass+'_state_dict.pt')
                    
                writer.add_scalar("Loss/valid", valid_total_loss/len(valid_loader), epoch)
                writer.add_scalar("Acc/valid", valid_total_acc/len(valid_data), epoch)
                writer.flush()
                print(f"valid_loss:{valid_total_loss/len(valid_loader)}")
                print(f"valid_acc:{valid_total_acc/len(valid_data)}")


def vanilla_test(
    model: nn.Module,
    test_data: Dataset,
    test_loader: DataLoader,
    device: torch.device,
    pad_id: int = 0,
    sep_id: int = None,
    load_pass = None
):

    if load_pass is not None:
        model.load_state_dict(torch.load(load_pass)) 

    model.eval()
    with torch.no_grad():
        test_total_loss = 0.0
        test_total_acc = 0.0
        loss_rec_inst = ReconstructionLoss()
            
        for x, true_y in test_loader:
            x = x.to(device)
            true_y = true_y.to(device)
            tgt_mask = make_tgt_mask(true_y.shape[1]).to(device)
            src_key_padding_mask = (x == pad_id)
            tgt_key_padding_mask = (true_y == pad_id)

            pred_y = model(x, true_y, tgt_mask=tgt_mask,
                                 src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

            loss = loss_rec_inst(pred_y,true_y)
            test_total_loss += loss.item()
            test_total_acc += calculate_acc(pred_y, true_y,pad_id=pad_id)

            
        print(f"test_loss:{test_total_loss/len(test_loader)}")
        print(f"test_acc:{test_total_acc/len(test_data)}")


def vanilla_print_sample(
    x: torch.Tensor,
    true_y: torch.Tensor,
    id2word_dic,
    model: nn.Module,
    device: torch.device,
    pad_id: int = 0,
    sep_id: int = None,
    load_pass = None
):
    """[summary]

    Args:
        x (torch.Tensor): [description]
        true_y (torch.Tensor): [description]
        id2word_dic ([type]): [description]
        model (nn.Module): [description]
        device (torch.device): [description]
        pad_id (int, optional): [description]. Defaults to 0.
        sep_id (int, optional): [description]. Defaults to None.

        x:['<CLS>', 'Z', '=', '1', ',', 'A', '=', 'Z', '+', '2', ',', 'H', '=', '1', '+', 'A', '<SEP>', 'H', '=', '<SEP>']
        true_y:['<CLS>', '4', '<SEP>', '<PAD>']
        pred_y:['<CLS>', '4', '<SEP>', '<SEP>']
    """

    if load_pass is not None:
        model.load_state_dict(torch.load(load_pass)) 

    for x_item,true_y_item in zip(x,true_y):
        x_item = torch.unsqueeze(x_item.to(device), 0)
        true_y_item = torch.unsqueeze(true_y_item.to(device), 0)
        print(f"x:{[id2word_dic[id] for id in x_item[0].tolist()]}")
        print(f"true_y:{[id2word_dic[id] for id in true_y_item[0].tolist()]}")

        tgt_mask = make_tgt_mask(true_y_item.shape[1]).to(device)
        src_key_padding_mask = (x_item == pad_id)
        tgt_key_padding_mask = (true_y_item == pad_id)

        pred_y = model(x_item, true_y_item, tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        print(
            f"pred_y:{['<CLS>']+[id2word_dic[id] for id in torch.argmax(pred_y[0],dim=-1)[:-1].tolist()]}")


