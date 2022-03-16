import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.train_utils import *
from dataset import PCN_pcd
import h5py


def save_h5(data, path):
    f = h5py.File(path, 'w')
    a = data.data.cpu().numpy()
    f.create_dataset('data', data=a)
    f.close()

def save_obj(point, path):
    n = point.shape[0]
    with open(path, 'w') as f:
        for i in range(n):
            f.write("v {0} {1} {2}\n".format(point[i][0],point[i][1],point[i][2]))
    f.close()

def test():
    dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    test_loss_cat = torch.zeros([8, 4], dtype=torch.float32).cuda()
    cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 150
    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft']

    logging.info('Testing...')

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):

            label, inputs_cpu, gt_cpu, obj = data

            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            result_dict = net(inputs, gt, is_training=False)
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            for j, l in enumerate(label):
                for ind, m in enumerate(metrics):
                    test_loss_cat[int(l), ind] += result_dict[m][int(j)]

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            if args.save_vis:
                for j in range(args.batch_size):
                    if not os.path.isdir(os.path.join(os.path.dirname(args.load_model), 'all', str(label[j]))):
                        os.makedirs(os.path.join(os.path.dirname(args.load_model), 'all', str(label[j])))
                        
                    path = os.path.join(os.path.dirname(args.load_model), 'all', str(label[j]), str(obj[j])+'.obj')
                    save_obj(result_dict['out2'][j], path)

        logging.info('Loss per category:')
        category_log = ''
        for i in range(8):
            category_log += '\ncategory name: %s' % (cat_name[i])
            for ind, m in enumerate(metrics):
                scale_factor = 1 if m == 'f1' else 10000
                category_log += ' %s: %f' % (m, test_loss_cat[i, ind] / cat_num[i] * scale_factor)
        logging.info(category_log)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = os.path.join('./cfgs',arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()
