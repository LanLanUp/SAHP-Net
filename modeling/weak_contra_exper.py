import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from semi_UNet import semi_Unet
from semi_UNet64 import semi_Unet64
from max_UNet import max_Unet
from prototype_UNet import proto_Unet
from prototype_UNet64 import proto_Unet64
from attention_unet import AttU_Net
from unet import Unet
from UNet_uncrf import Unet_uncrf

from metrics import *
from contra_dataset import *
from utils import *

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
import math
from plot import loss_plot
from plot import metrics_plot

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gpu', type=str, default='1')
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
    parse.add_argument("--epoch", type=int, default=200)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNet_uncrf',
                       help='semi_UNet/semi_UNet64/max_UNet/proto_UNet/proto_UNet64/attention_unet/UNet/UNet_uncrf')
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument('--dataset', default='TN3K',  # dsb2018_256
                       help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung/Lidc/Thyroid')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    parse.add_argument("--num_slide",type=int,default=12)
    parse.add_argument("--pretrain",type=str,default='False')#'True'
    parse.add_argument("--coefficient", type=float, default=0.3)
    parse.add_argument('-criterion', type=str, default='Dice')
    parse.add_argument('-recent_data', type=str, default='TN3K')
    args = parse.parse_args()
    return args

def getLog(args):
    #dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    dirname = '/home/lingeng/weak_supervise/result/log/'+args.arch+'/'+args.recent_data+'/coeffi_'+str(args.coefficient) + '/1000_proj_only'
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):

    if args.arch == 'semi_UNet':
        model = semi_Unet(1,1).to(device)
    if args.arch == 'max_UNet':
        model = max_Unet(1,1).to(device)
    if args.arch == 'proto_UNet':
        model = proto_Unet(1,1).to(device)
    if args.arch == 'proto_UNet64':
        model = proto_Unet64(1,1).to(device)
    if args.arch == 'semi_UNet64':
        model = semi_Unet64(1,1).to(device)
    if args.arch == 'attention_unet':
        model = AttU_Net(1,1).to(device)
    if args.arch == 'UNet':
        model = Unet(1,1).to(device)
    if args.arch == 'UNet_uncrf':
        model = Unet_uncrf(1,1).to(device)


    if args.pretrain == 'True':#加载之前没有训练完成的模型
        model.load_state_dict(torch.load(r'/home/lingeng/weak_supervise/saved_model/proto_UNet64/weak_sat_0.5/coeffi_0.15/200_5176.pth'))
        print('successfully load the model!')
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset == "TN3K":#thyroid /home/zelan/p/review_code/thyroid
        train_dataset = TrainDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = ValDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        #test_dataloaders = val_dataloaders
        test_dataset = ValDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model,args,best_iou,val_dataloaders):
    # print('hihi')
    model = model.eval()
    with torch.no_grad():
        i=0   #the i picture in val dataset
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  #the sum images number
        #for ii, sample_batched in enumerate(val_dataloaders):
            #x, y  = sample_batched['images'].cuda(),
        start_t = time.time()

        for x,lab,pic_path,mask_path in val_dataloaders:
            # fore_mask = fore_mask.to(device)
            # back_mask = back_mask.to(device)
            lab = lab.to(device)
            x = x.to(device)  # orignal picture
            #print(x.type)#这一行是出现那么多type的原因
            # y = model(x)#torch计算很快，1*1*256*256
            _,y = model(x, lab, lab)
            # out_x,out_y,y,out_proto = model(x,recti_label,recti_label)
            #print('x',x.shape)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                #print(y.type,y.shape,y)
                img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要压缩一维表示batchsize，图片处理很快
                # pre_x = torch.squeeze(norm_x).cpu().numpy()
                # pre_y = torch.squeeze(norm_y).cpu().numpy()
                # lab_x = torch.squeeze(lab_y).cpu().numpy()
                # lab_y = torch.squeeze(lab_y).cpu().numpy()


            tem_iou = get_iou(mask_path[0], img_y)
            # x_iou = xy_iou(lab_x,pre_x)
            # y_iou = xy_iou(lab_y, pre_y)
            # tem_iou = (x_iou + y_iou) / 2
            if math.isnan(tem_iou):
                tem_iou = 0
            miou_total += tem_iou
            if i < num: i += 1
        print('测试一轮花费的时间：',time.time()-start_t)
        aver_iou = miou_total / num
        print('Miou=',aver_iou)
        logging.info('Miou:{}'.format(aver_iou))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            save_dire = './saved_model/'+str(args.arch)+'/'+args.recent_data+'/coeffi_'+str(args.coefficient)+'/'
            if not os.path.exists(save_dire):
                os.makedirs(save_dire)
            #torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
            torch.save(model.state_dict(), save_dire + str(args.batch_size)+'_'+str(args.epoch)+'.pth')
            '''torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       r'./saved_model/' + str(args.dataset) +'/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(
                           args.dataset) + '_' + str(args.epoch) + '.pth')'''
        print(f'highest_iou:{best_iou}')
        return best_iou, aver_iou

def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_iou,aver_iou,aver_dice,aver_hd = 0,0,0,0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for img,fore_mask,back_mask in train_dataloaders:
            step += 1
            inputs = img.to(device)
            fore_masks = fore_mask.to(device)
            back_masks = back_mask.to(device)
            # labels_x = lab_x.to(device)
            # labels_y = lab_y.to(device)
            # rec_labels = rec_label.to(device)

            optimizer.zero_grad()

            unc_max_rect,out_argmax = model(inputs,fore_masks,back_masks)
            loss1 = criterion(out_argmax,fore_masks)
            loss2 = criterion(unc_max_rect,out_argmax)
            loss = loss1 + args.coefficient * loss2
            # loss_norm_x = criterion(norms_x,labels_x)#在这一行报错
            # loss_norm_y = criterion(norms_y, labels_y)
            # loss_norm = loss_norm_x + loss_norm_y
            # loss_proto_x = criterion(protos_x,labels_x)
            # loss_proto_y = criterion(protos_y, labels_y)
            # loss_proto = loss_proto_x + loss_proto_y
            # loss_norm = criterion(out_norms,fore_masks)
            # # loss_proto = criterion(out_protos,fore_masks)
            # loss1 = loss_norm
            # loss2 = criterion(out_norms,out_protos)
            # # loss =  loss1 + args.coefficient * loss2
            # loss = loss1 + args.coefficient * loss2
            # # outputs = model(inputs)
            # # loss = criterion(fore_masks,outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item() / 2.1))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))

        print('epoch_loss:',epoch_loss)
        loss_list.append(epoch_loss)
        writer.add_scalar('epoch_loss', epoch_loss, global_step=epoch)
        best_iou, aver_iou = val(model, args, best_iou, val_dataloader)
        writer.add_scalar('aver_iou', aver_iou, global_step=epoch)
        iou_list.append(aver_iou)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    with open('/home/lingeng/weak_supervise/result/best_iou', 'a+') as file:
        file.write(args.arch+'_'+str(args.epoch)+'的最高iou值为：'+str(best_iou)+'\n')
    loss_plot(args, loss_list,'loss')
    loss_plot(args, iou_list,'iou')
    return model

def test(val_dataloaders,args,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        # dir = os.path.join(r'./predict_mask',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        coeffi_dir = 'coeffi_'+str(args.coefficient)
        dir = './predict_mask/' + str(args.arch) + '/' + args.recent_data + '/' + str(coeffi_dir) + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')

    # model.load_state_dict(torch.load(r'/home/lingeng/unetzoo-tn3k/saved_model/tn3k/myChannelUnet_16_tn3k_40.pth', map_location='cpu'))
    model.load_state_dict(
        torch.load(r'/home/lingeng/weak_supervise/saved_model/UNet_uncrf/tg3k_sat_0.5/coeffi_0.3/15_5643.pth'))
    model.eval()

    #plt.ion() #开启动态模式
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae'])
        for pic,lab,pic_path,mask_path in test_dataloaders:
            pic = pic.to(device)
            lab = lab.to(device)
            _,predict = model(pic,lab,lab) #1*1*256*256，半监督测试
            # predict = model(pic)#unet全监督测试
            save_png = torch.squeeze(predict[-1]).cpu().numpy()
            save_png = (save_png >0.5).astype('float') * 255
            save_png = save_png.astype(np.uint8)
            save_png = Image.fromarray(save_png)
            save_png.save(dir + pic_path[0].split('/')[-1])
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #predict 256*256
            _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae = evaluate(mask_path[0], predict)
            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, auc=_auc)
        metrics_result = metrics.mean(len(test_dataloaders))
        print("Test Result:")
        print(
            'recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, auc: %.4f'
            % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
               metrics_result['F1_score'],
               metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
               metrics_result['auc']))
        # evaluation_dir = os.path.sep.join(['result', 'metrics', 'test-tn3k' + '/'])
        evaluation_dir = './result/' + 'test-result/' + str(args.arch) + '/' + str(args.recent_data) + '/coeffi_' + str(args.coefficient) + '/'
        if not os.path.exists(evaluation_dir):
            os.makedirs(evaluation_dir)
        values_txt = '0' + '\t'
        for k, v in metrics_result.items():
            v = 100 * v
            # keys_txt += k + '\t'
            values_txt += '%.2f' % v + '\t'
        text = values_txt + '\n'
        # save_path = evaluation_dir + args.arch + '.txt'
        save_path = evaluation_dir + 'metrics.txt'
        with open(save_path, 'a+') as f:
            f.write(text)
        print(f'metrics saved in {save_path}')
        print("------------------------------------------------------------------")



if __name__ =="__main__":
    time_begin = time.time()
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        #transforms.Normalize([0.5], [0.5])  # ->[-1,1]
    ])
    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()
    #
    f = torch.cuda.is_available()
    device = torch.device("cuda" if f else "cpu")
    # device = torch.device("cpu")
    print(device)
    args = getArgs()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    writer_dir = os.path.join(r'runs/logs',str(args.arch)+'_'+str(args.dataset))
    writer = SummaryWriter(writer_dir)

    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    if args.criterion == 'Dice':
        criterion = soft_dice
    else:
        criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(model.parameters(),lr=0.0001)


    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders, val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, args,save_predict=True)
    print('训练总共花费的时间为：',time.time()-time_begin)
