import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, SVHN
import torch.nn.functional as F

import numpy as np
import random
import matplotlib.pyplot as plt
import wandb
import argparse

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from siamese_resnet import BasicBlock, Siamese_ResNet     
from useful_fc import show_auroc


class CIFAR10_TestPair(Dataset):
    def __init__(
            self,
            ood_dataset='cifar10_test',  # 'cifar10_test', 'tinyimagenet', 'lsun', 'gaussian'
            K=5,
            N=6,
            n_query_img=1000,
            seen_classes=list(range(6)),
            ood_classes=[8, 9],
            seed=2021
    ):
        self.K = K
        self.N = N
        self.batch_size = K * N
        self.n_query_img = n_query_img
        self.seen_classes = seen_classes
        self.ood_classes = ood_classes
        self.seed = seed
        self.seen_dataset = cifar10_dataset_test
        self.seen_class_dict = {k: [] for k in seen_classes}
        for class_id in seen_classes:
            self.seen_class_dict[class_id] = [idx for idx, item in enumerate(self.seen_dataset) if item[1] == class_id]

            # ood dataset
        self.ood_dataset = ood_dataset
        if self.ood_dataset != 'gaussian':
            if self.ood_dataset == 'cifar10_test':
                self.ood_dataset = cifar10_dataset_test
                self.ood_class_dict = {k: [] for k in ood_classes}
                for class_id in ood_classes:
                    self.ood_class_dict[class_id] = [idx for idx, item in enumerate(self.ood_dataset) if
                                                     item[1] == class_id]
            if self.ood_dataset == 'tinyimagenet':
                self.ood_dataset = tinyimagenet_dataset
                self.ood_classes = [999]
                self.ood_class_dict = {999: list(range(len(tinyimagenet_dataset)))}
            if self.ood_dataset == 'lsun':
                self.ood_dataset = lsun_dataset
                self.ood_classes = [999]
                self.ood_class_dict = {999: list(range(len(lsun_dataset)))}

    def __len__(self) -> int:
        return self.n_query_img * self.batch_size

    def __getitem__(self, index: int):
        query_image = None
        support_image = None
        query_class = None
        support_class = None
        if_diff = None  # if from same class
        if_ood = None  # if the query image is from ood classes

        batch_id = index // self.batch_size
        pair_id = index % self.batch_size  # 0,...,29
        way_id = pair_id // self.K  # 0,...,5
        shot_id = pair_id % self.K  # 0,...,4

        # query image
        if batch_id % 2 == 1:
            # query image from ood classes
            if_ood = 1
            if self.ood_dataset != 'gaussian':
                random.seed(self.seed + batch_id)  # to ensure the same query image is selected within a batch
                query_class = random.choice(self.ood_classes)
                q_img_id = random.choice(self.ood_class_dict[query_class])
                query_image = self.ood_dataset[q_img_id][0]
            # gassian ood
            else:
                query_class = 999
                torch.manual_seed(self.seed + batch_id)
                query_image = torch.randn(3, 32, 32) + 0.5
                query_image = torch.clamp(query_image, 0, 1)
                query_image[0][0] = (query_image[0][0] - 0.4914) / 0.2471
                query_image[0][1] = (query_image[0][1] - 0.4822) / 0.2435
                query_image[0][2] = (query_image[0][2] - 0.4465) / 0.2616

        else:
            # query image from seen classes
            if_ood = 0
            random.seed(self.seed + batch_id)  # to ensure the same query image is selected within a batch
            query_class = random.choice(self.seen_classes)
            q_img_id = random.choice(self.seen_class_dict[query_class])
            query_image = self.seen_dataset[q_img_id][0]

        # support image:
        support_class = self.seen_classes[way_id]
        random.seed(self.seed + index)
        s_img_id = random.choice(self.seen_class_dict[support_class])
        support_image = self.seen_dataset[s_img_id][0]

        if_diff = int(query_class != support_class)

        query_class = torch.from_numpy(np.array([query_class], dtype=np.int))
        support_class = torch.from_numpy(np.array([support_class], dtype=np.int))
        if_diff = torch.from_numpy(np.array([if_diff], dtype=np.int))
        if_ood = torch.from_numpy(np.array([if_ood], dtype=np.int))

        return query_image, support_image, query_class, support_class, if_diff, if_ood


def test_evaluation(model, k_shot=5, n_way=6, n_query_img=1000, seen_classes=list(range(6)), ood_classes=[8, 9],
                    ood_dataset='cifar10_test', seed=2021, plot=True):
    if n_way == 1:
        print("skip n_way = 1 for now")
        query_img_stat = {"q_class": [], "q_is_ood": [], "q_pred_class": [],
                          "q_ood_score": [], "q_ideal_ood_score": []}
        test_metrics = {"auroc_query_is_ood": None, "auroc_query_is_ood_ideal": None, "accuracy_query_seen": None}

        return query_img_stat, test_metrics


    # data loading
    testset = CIFAR10_TestPair(ood_dataset, k_shot, n_way, n_query_img, seen_classes, ood_classes, seed)
    test_dataloader = DataLoader(
        testset,
        shuffle=False,  # shuffle = false to ensure the N-way K-shot order
        batch_size=k_shot * n_way,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    # train the logistic classifier
    model.eval()

    q_class = []
    q_is_ood = []
    q_pred_class = []
    q_ood_score = []
    q_ideal_ood_score = []

    for i, test_batch in enumerate(test_dataloader, 0):
        test_query_image, test_support_image, test_query_class, test_support_class, test_is_diff, test_is_ood = test_batch

        # getting embbeding and ideal max softmax if using pretrained linear layer
        support_emb = model(test_support_image.cuda(), test_support_image.cuda())[0].detach().cpu().numpy()    
        support_class = test_support_class.numpy().squeeze()

        query_emb = model(test_query_image[0].unsqueeze(0).cuda(), test_query_image[0].unsqueeze(0).cuda())[0]   
        query_emb = query_emb.detach().cpu().numpy()
        query_class = test_query_class.numpy().squeeze()[0]

        # fit multimonial logistic regression on support set
        clf = LogisticRegression(penalty='l2',
                                 random_state=seed,
                                 C=1.0,
                                 solver='lbfgs',
                                 max_iter=1000,
                                 multi_class='multinomial')
        clf.fit(support_emb, support_class)

        # get softmax score of query image
        query_softmax = clf.predict_proba(query_emb)
        query_class_pred = clf.predict(query_emb)[0]

        # use 1-max_softmax as ood score
        q_class.append(query_class)
        q_is_ood.append(test_is_ood[0].cpu().detach().numpy()[0])
        q_pred_class.append(query_class_pred)
        q_ood_score.append(1 - query_softmax.max())

        query_img_stat = {"q_class": q_class, "q_is_ood": q_is_ood, "q_pred_class": q_pred_class,
                          "q_ood_score": q_ood_score, "q_ideal_ood_score": q_ideal_ood_score}

    # evaluation metrics
    test_metrics = {}

    print("distribution of test ood score")
    test_seen_score = q_ood_score[0::2]
    test_ood_score = q_ood_score[1::2]

    if plot:
        plt.hist(test_seen_score, alpha=0.5, bins=20, color='g', label='seen')
        plt.hist(test_ood_score, alpha=0.5, bins=20, color='b', label='ood')
        plt.legend()
        plt.show()

    test_metrics["auroc_query_is_ood"] = show_auroc(q_is_ood, q_ood_score, display=plot)
    print("auroc of query image ood score: ", test_metrics["auroc_query_is_ood"])


    test_metrics["accuracy_query_seen"] = metrics.accuracy_score(q_class[0::2], q_pred_class[0::2])
    print("accuracy of query image from seen classes:", test_metrics["accuracy_query_seen"])

    return query_img_stat, test_metrics



if __name__ == "__main__":
    wandb.finish()

    ### argparse ###
    parser = argparse.ArgumentParser()
    parser.add_argument("k_shot", type=int)
    parser.add_argument("n_way", type=int)
    parser.add_argument("n_query_image", type=int)
    parser.add_argument("ood_dataset")
    parser.add_argument("seed", type=int)
    parser.add_argument("exper")
    args = parser.parse_args()
    print('run {}-shot {}-way {} - {}'.format(args.k_shot, args.n_way, args.ood_dataset, args.exper))

    ### load datasets ###
    transform = T.Compose([T.ToTensor(),
                           T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)), ])
    cifar10_dataset = CIFAR10(root="./data", train=True, transform=transform, download=False)
    torch.manual_seed(2021)
    cifar10_dataset_train, cifar10_dataset_val = torch.utils.data.random_split(cifar10_dataset, [40000, 10000])
    cifar10_dataset_test = CIFAR10(root="./data", train=False, transform=transform, download=False)
    svhn_dataset = SVHN(root="./data", split='train', transform=transform, download=False)
    tinyimagenet_dataset = torchvision.datasets.ImageFolder("./data/Imagenet_resize", transform=transform)
    lsun_dataset = torchvision.datasets.ImageFolder("./data/LSUN_resize", transform=transform)
    print("load datasets finished")


    ### load model ###
    model = Siamese_ResNet(BasicBlock, [2,2,2,2]).cuda() 
    model.load_state_dict(torch.load("./stat_dict/siamese_without_oe_best_rotation.pth"))   
    #model.load_state_dict(torch.load("./stat_dict/baseline_pretrain_with_oe.pth"))
    model.eval()
    print("load model finished")

    ### test evaluation and log to wandb ###
    wandb.login()
    wandb.init(mode="online",
               name='{}-shot {}-way {} - {}'.format(args.k_shot, args.n_way, args.ood_dataset, args.exper),
               project = 'Testing Evaluation - Siamese_Max_Softmax',  
               tags=['{}-shot'.format(args.k_shot), '{}-way'.format(args.n_way), args.ood_dataset, args.exper],
               config={"k_shot": args.k_shot,
                       "n_way": args.n_way,
                       "n_query_image": args.n_query_image,
                       "seen_classes": list(range(10-args.n_way, 10)), #list(range(args.n_way)),
                       "ood_classes": ([0,1] if args.ood_dataset == 'cifar10_test' else [999]), #([8,9] if args.ood_dataset == 'cifar10_test' else [999]),
                       "ood_dataset": args.ood_dataset,
                       "seed": args.seed
                       })

    query_img_stat, test_metrics = test_evaluation(
        model,
        wandb.config.k_shot,
        wandb.config.n_way,
        wandb.config.n_query_image,
        wandb.config.seen_classes,
        wandb.config.ood_classes,
        wandb.config.ood_dataset,
        wandb.config.seed,
        plot=False)

    wandb.log({
        "auroc_query_is_ood": test_metrics['auroc_query_is_ood'],
        "accuracy_query_seen": test_metrics['accuracy_query_seen']
    })

    wandb.finish()
    print("wandb logging finished")

