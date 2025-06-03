import pandas as pd
import sklearn.cluster
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import torch




# 最基本的指标
def resulting0(ground_truth, y_train_pred):
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(y_train_pred, torch.Tensor):
        y_train_pred = y_train_pred.cpu().numpy()

    ACC = metrics.accuracy_score(ground_truth, y_train_pred)   # 宏平均

    fpr, tpr, thresholds = metrics.roc_curve(ground_truth,y_train_pred)
    AUC = metrics.auc(fpr, tpr)
    y_pred = np.where(np.array(y_train_pred) > 0.5, 1, 0)
    tn, fp, fn, tp = metrics.confusion_matrix(ground_truth, y_pred, labels=[0, 1]).ravel()
    SPE = tn / (tn + fp)
    SEN = tp / (tp + fn)
    PRE = tp / (tp + fp)
    f1_score = 2 * PRE * SEN / (PRE + SEN)


    print('TN:', tn, ' TP:', tp, ' FP:', fp, ' FN:', fn, ' ACC:',ACC,' SPE:',SPE, ' SEN:',SEN, ' PRE:',PRE, ' AUC:',AUC, ' F1:',f1_score)



# 最基本的指标 + 判对恶性（判对总恶性 + 判对原恶性） 
def resulting1(names, ground_truth, y_train_pred):

    ACC = metrics.accuracy_score(ground_truth, y_train_pred)   # 宏平均

    fpr, tpr, thresholds = metrics.roc_curve(ground_truth,y_train_pred)
    AUC = metrics.auc(fpr, tpr)
    y_pred = np.where(np.array(y_train_pred) > 0.5, 1, 0)
    tn, fp, fn, tp = metrics.confusion_matrix(ground_truth, y_pred, labels=[0, 1]).ravel()
    SPE = tn / (tn + fp)
    SEN = tp / (tp + fn)
    PRE = tp / (tp + fp)
    f1_score = 2 * PRE * SEN / (PRE + SEN)

    
    # 输出判断对几个恶性
    # 原恶性列表
    exing = ['C19004A_0003627-Anonymized-202205231529-D-L.dcm',
             'C19007A_0000109-Anonymized-202009030920-D-L.dcm',
             'C20001D_0005899-Anonymized-202302280958-D-L.dcm',
             'C20001D_0002870-Anonymized-202108061002-D-L.dcm',
             'C20001D_0002593-Anonymized-202107271523-D-L.dcm',
             'C20001D_0000121-Anonymized-202104221406-D-L.dcm',
             'C19005B_0000346-Anonymized-202105281616-D-R.dcm',
             'C19004A_0004570-Anonymized-202211041453-D-R.dcm',
             'C20001D_0003937-Anonymized-202110191404-D-L.dcm',
             'C19005B_0000492-Anonymized-202106081343-D-R.dcm',
             'C19004A_0002584-Anonymized-202107081641-D-L.dcm',
             'C18002F_0000048-Anonymized-202305151448-D-L.dcm',
             'C18005D_0000209-Anonymized-202207191518-D-R.dcm',
             'C20005B_0000034-Anonymized-202207051445-D-R.dcm'
             ]
    # 总恶性列表
    exing1 = ['C19004A_0003627-Anonymized-202205231529-D-L.dcm',
             'C19007A_0000109-Anonymized-202009030920-D-L.dcm',
             'C20001D_0005899-Anonymized-202302280958-D-L.dcm',
             'C20001D_0002870-Anonymized-202108061002-D-L.dcm',
             'C20001D_0002593-Anonymized-202107271523-D-L.dcm',
             'C20001D_0000121-Anonymized-202104221406-D-L.dcm',
             'C19005B_0000346-Anonymized-202105281616-D-R.dcm',
             'C19004A_0004570-Anonymized-202211041453-D-R.dcm',
             'C20001D_0003937-Anonymized-202110191404-D-L.dcm',
             'C19005B_0000492-Anonymized-202106081343-D-R.dcm',
             'C19004A_0002584-Anonymized-202107081641-D-L.dcm',
             'C18002F_0000048-Anonymized-202305151448-D-L.dcm',
             'C18005D_0000209-Anonymized-202207191518-D-R.dcm',
             'C20005B_0000034-Anonymized-202207051445-D-R.dcm',
             '010-BJBA-00010-WHL-201709281420-D.dcm',
            '021-SHZL-00043-SHWY-201708280835-D.dcm',
            '021-SHZL-00055-SLH-201709221514-D.dcm',
            '021-SHZL-00065-CYZH-201710171000-D.dcm',
            '021-SHZL-00074-XXX-201711011518-D.dcm',
            '021-SHZL-00085-ZHJ-201711081326-D.dcm',
            '021-SHZL-00105-LCHL-201712201356-D.dcm',
            '021-SHZL-00127-ZHJH-201802260946-D.dcm',
            '021-SHZL-00150-FPP-201804101358-D.dcm',
            '021-SHZL-00200-FYS-201806211142-D.dcm',
            '021-SHZL-00208-ZHRM-201806260849-D-L.dcm',
            '021-SHZL-00225-JJJ-201807050957-D.dcm',
            '021-SHZL-00269-JGY-201808060901-D.dcm',
            '021-SHZL-00271-LTL-201808061024-D.dcm',
            '021-SHZL-00358-CHXF-201811010834-D.dcm',
            '021-SHZL-00396-CHGM-201812291147-D.dcm',
            '021-SHZL-00397-LF-2019-01041348-D.dcm',
            '021-SHZL-00400-CYF-201901141412-D.dcm',
            '021-SHZL-00441-FZH-201906271129-D.dcm',
            '021-SHZL-00449-JJF-201907030928-D.dcm',
            '021-SHZL-00459-ZHXH-201907311404-D.dcm',
            '022-TJZL-00003-LJM-201705311617-D.dcm',
            '022-TJZL-00017-SWF-201711141040-D.dcm',
            '022-TJZL-00019-MZHH-201712111450-D-L.dcm',
            '022-TJZL-00024-MGX-201712250936-D.dcm',
            '023-CHQSY-00002-HNZH-201804032145-D.dcm',
            '023-CHQSY-00013-YJH-201807051013-D.dcm',
            '023-CHQSY-00016-CHSHH-201807300857-D.dcm',
            '023-CHQSY-00038-ZHYP-201904021333-D.dcm',
            '029-XAJD-00009-MYF-201805070957-D.dcm',
            '029-XAJD-00026-XQN-201805150934-D.dcm',
            '029-XAJD-00040-YY-201805171129-D.dcm',
            '029-XAJD-00140-WRR-201807231515-D.dcm',
            '029-XAJD-00178-ZHMD-201809120940-D.dcm',
            '029-XAJD-00233-LSHF-201901101412-D.dcm',
            '029-XAJD-00256-FD-201905291444-D.dcm',
            '029-XAJD-00308-SHLL-201911011050-D.dcm',
            '029-XAJD-00309-SHLL-201911011059-D.dcm',
            '0531-SDZL-00004-SHY-201809050802-D.dcm',
            '0531-SDZL-00015-WYQQG-201810261447-D.dcm',
            '0531-SDZL-00056-LPZH-201905220850-D.dcm',
            '0531-SDZL-00062-CHCHS-201906101629-D.dcm',
            '0531-SDZL-00079-HXL-201910081126-D.dcm',
            '0571-ZHJSH-00010-XNQ-201807031418-D.dcm',
            '0571-ZHJSH-00051-ZHAF-201807190951-D-L.dcm',
            '0571-ZHJSH-00304-WJF-201905201607-D.dcm',
            '0571-ZHJSH-00319-WFH-201906051511-D.dcm',
            '0571-ZHJSH-00351-ZHYF-201906190937-D.dcm',
            '0571-ZHJSH-00415-HLH-201907171034-D.dcm'
             ]
    correct_tumor = []
    correct_origin_exing = []
    for i in range(len(names)):
        if ground_truth[i] == 1 and y_train_pred[i] == ground_truth[i]:
            if names[i] in exing1:
                correct_tumor.append(names[i])
            if names[i] in exing:
                correct_origin_exing.append(names[i])


    print('TN:', tn, ' TP:', tp, ' FP:', fp, ' FN:', fn, ' ACC:',ACC,' SPE:',SPE, ' SEN:',SEN, ' PRE:',PRE, ' AUC:',AUC, ' F1:',f1_score, ' 判对恶性：', len(correct_tumor), '(', len(correct_origin_exing), ')')



# 最基本的指标 + 判对恶性（判对总恶性 + 判对原恶性） + 判对哪些恶性 判对哪些良性
def resulting2(name, names, ground_truth, y_train_pred, is_positive = 0):
    ACC = metrics.accuracy_score(ground_truth, y_train_pred)   # 宏平均

    fpr, tpr, thresholds = metrics.roc_curve(ground_truth,y_train_pred)
    AUC = metrics.auc(fpr, tpr)
    y_pred = np.where(np.array(y_train_pred) > 0.5, 1, 0)
    tn, fp, fn, tp = metrics.confusion_matrix(ground_truth, y_pred, labels=[0, 1]).ravel()
    SPE = tn / (tn + fp)
    SEN = tp / (tp + fn)
    PRE = tp / (tp + fp)
    f1_score = 2 * PRE * SEN / (PRE + SEN)


    # print('ACC:',ACC)
    # print('SPE:',SPE)
    # print('SEN:',SEN)
    # print('PRE:',PRE)
    # print('AUC:',AUC)
    # print('F1:',f1_score)


    exing = ['C19004A_0003627-Anonymized-202205231529-D-L.dcm',
             'C19007A_0000109-Anonymized-202009030920-D-L.dcm',
             'C20001D_0005899-Anonymized-202302280958-D-L.dcm',
             'C20001D_0002870-Anonymized-202108061002-D-L.dcm',
             'C20001D_0002593-Anonymized-202107271523-D-L.dcm',
             'C20001D_0000121-Anonymized-202104221406-D-L.dcm',
             'C19005B_0000346-Anonymized-202105281616-D-R.dcm',
             'C19004A_0004570-Anonymized-202211041453-D-R.dcm',
             'C20001D_0003937-Anonymized-202110191404-D-L.dcm',
             'C19005B_0000492-Anonymized-202106081343-D-R.dcm',
             'C19004A_0002584-Anonymized-202107081641-D-L.dcm',
             'C18002F_0000048-Anonymized-202305151448-D-L.dcm',
             'C18005D_0000209-Anonymized-202207191518-D-R.dcm',
             'C20005B_0000034-Anonymized-202207051445-D-R.dcm',
             'C19004A_0003627-Anonymized-202205231529-D-L-00-000000.dcm',
            'C19007A_0000109-Anonymized-202009030920-D-L-00-000000.dcm',
            'C20001D_0005899-Anonymized-202302280958-D-L-00-000000.dcm',
            'C20001D_0002870-Anonymized-202108061002-D-L-00-000000.dcm',
            'C20001D_0002593-Anonymized-202107271523-D-L-00-000000.dcm',
            'C20001D_0000121-Anonymized-202104221406-D-L-00-000000.dcm',
            'C19005B_0000346-Anonymized-202105281616-D-R-00-000000.dcm',
            'C19004A_0004570-Anonymized-202211041453-D-R-00-000000.dcm',
            'C20001D_0003937-Anonymized-202110191404-D-L-00-000000.dcm',
            'C19005B_0000492-Anonymized-202106081343-D-R-00-000000.dcm',
            'C19004A_0002584-Anonymized-202107081641-D-L-00-000000.dcm',
            'C18002F_0000048-Anonymized-202305151448-D-L-00-000000.dcm',
            'C18005D_0000209-Anonymized-202207191518-D-R-00-000000.dcm',
            'C20005B_0000034-Anonymized-202207051445-D-R-00-000000.dcm',
             ]
        # 总恶性列表
    exing1 = ['C19004A_0003627-Anonymized-202205231529-D-L.dcm',
             'C19007A_0000109-Anonymized-202009030920-D-L.dcm',
             'C20001D_0005899-Anonymized-202302280958-D-L.dcm',
             'C20001D_0002870-Anonymized-202108061002-D-L.dcm',
             'C20001D_0002593-Anonymized-202107271523-D-L.dcm',
             'C20001D_0000121-Anonymized-202104221406-D-L.dcm',
             'C19005B_0000346-Anonymized-202105281616-D-R.dcm',
             'C19004A_0004570-Anonymized-202211041453-D-R.dcm',
             'C20001D_0003937-Anonymized-202110191404-D-L.dcm',
             'C19005B_0000492-Anonymized-202106081343-D-R.dcm',
             'C19004A_0002584-Anonymized-202107081641-D-L.dcm',
             'C18002F_0000048-Anonymized-202305151448-D-L.dcm',
             'C18005D_0000209-Anonymized-202207191518-D-R.dcm',
             'C20005B_0000034-Anonymized-202207051445-D-R.dcm',
             'C19004A_0003627-Anonymized-202205231529-D-L-00-000000.dcm',
            'C19007A_0000109-Anonymized-202009030920-D-L-00-000000.dcm',
            'C20001D_0005899-Anonymized-202302280958-D-L-00-000000.dcm',
            'C20001D_0002870-Anonymized-202108061002-D-L-00-000000.dcm',
            'C20001D_0002593-Anonymized-202107271523-D-L-00-000000.dcm',
            'C20001D_0000121-Anonymized-202104221406-D-L-00-000000.dcm',
            'C19005B_0000346-Anonymized-202105281616-D-R-00-000000.dcm',
            'C19004A_0004570-Anonymized-202211041453-D-R-00-000000.dcm',
            'C20001D_0003937-Anonymized-202110191404-D-L-00-000000.dcm',
            'C19005B_0000492-Anonymized-202106081343-D-R-00-000000.dcm',
            'C19004A_0002584-Anonymized-202107081641-D-L-00-000000.dcm',
            'C18002F_0000048-Anonymized-202305151448-D-L-00-000000.dcm',
            'C18005D_0000209-Anonymized-202207191518-D-R-00-000000.dcm',
            'C20005B_0000034-Anonymized-202207051445-D-R-00-000000.dcm',
             '010-BJBA-00010-WHL-201709281420-D.dcm',
            '021-SHZL-00043-SHWY-201708280835-D.dcm',
            '021-SHZL-00055-SLH-201709221514-D.dcm',
            '021-SHZL-00065-CYZH-201710171000-D.dcm',
            '021-SHZL-00074-XXX-201711011518-D.dcm',
            '021-SHZL-00085-ZHJ-201711081326-D.dcm',
            '021-SHZL-00105-LCHL-201712201356-D.dcm',
            '021-SHZL-00127-ZHJH-201802260946-D.dcm',
            '021-SHZL-00150-FPP-201804101358-D.dcm',
            '021-SHZL-00200-FYS-201806211142-D.dcm',
            '021-SHZL-00208-ZHRM-201806260849-D-L.dcm',
            '021-SHZL-00225-JJJ-201807050957-D.dcm',
            '021-SHZL-00269-JGY-201808060901-D.dcm',
            '021-SHZL-00271-LTL-201808061024-D.dcm',
            '021-SHZL-00358-CHXF-201811010834-D.dcm',
            '021-SHZL-00396-CHGM-201812291147-D.dcm',
            '021-SHZL-00397-LF-2019-01041348-D.dcm',
            '021-SHZL-00400-CYF-201901141412-D.dcm',
            '021-SHZL-00441-FZH-201906271129-D.dcm',
            '021-SHZL-00449-JJF-201907030928-D.dcm',
            '021-SHZL-00459-ZHXH-201907311404-D.dcm',
            '022-TJZL-00003-LJM-201705311617-D.dcm',
            '022-TJZL-00017-SWF-201711141040-D.dcm',
            '022-TJZL-00019-MZHH-201712111450-D-L.dcm',
            '022-TJZL-00024-MGX-201712250936-D.dcm',
            '023-CHQSY-00002-HNZH-201804032145-D.dcm',
            '023-CHQSY-00013-YJH-201807051013-D.dcm',
            '023-CHQSY-00016-CHSHH-201807300857-D.dcm',
            '023-CHQSY-00038-ZHYP-201904021333-D.dcm',
            '029-XAJD-00009-MYF-201805070957-D.dcm',
            '029-XAJD-00026-XQN-201805150934-D.dcm',
            '029-XAJD-00040-YY-201805171129-D.dcm',
            '029-XAJD-00140-WRR-201807231515-D.dcm',
            '029-XAJD-00178-ZHMD-201809120940-D.dcm',
            '029-XAJD-00233-LSHF-201901101412-D.dcm',
            '029-XAJD-00256-FD-201905291444-D.dcm',
            '029-XAJD-00308-SHLL-201911011050-D.dcm',
            '029-XAJD-00309-SHLL-201911011059-D.dcm',
            '0531-SDZL-00004-SHY-201809050802-D.dcm',
            '0531-SDZL-00015-WYQQG-201810261447-D.dcm',
            '0531-SDZL-00056-LPZH-201905220850-D.dcm',
            '0531-SDZL-00062-CHCHS-201906101629-D.dcm',
            '0531-SDZL-00079-HXL-201910081126-D.dcm',
            '0571-ZHJSH-00010-XNQ-201807031418-D.dcm',
            '0571-ZHJSH-00051-ZHAF-201807190951-D-L.dcm',
            '0571-ZHJSH-00304-WJF-201905201607-D.dcm',
            '0571-ZHJSH-00319-WFH-201906051511-D.dcm',
            '0571-ZHJSH-00351-ZHYF-201906190937-D.dcm',
            '0571-ZHJSH-00415-HLH-201907171034-D.dcm'
             ]
    correct_tumor = []
    correct_origin_exing = []
    for i in range(len(names)):
        if ground_truth[i] == 1 and y_train_pred[i] == ground_truth[i]:
            if names[i] in exing1:
                correct_tumor.append(names[i])
            if names[i] in exing:
                correct_origin_exing.append(names[i])



    print('TN:', tn, ' TP:', tp, ' FP:', fp, ' FN:', fn, ' ACC:',ACC,' SPE:',SPE, ' SEN:',SEN, ' PRE:',PRE, ' AUC:',AUC, ' F1:',f1_score, ' 判对恶性：', len(correct_tumor), '(', len(correct_origin_exing), ')')



    classes = list(set(ground_truth))
    classes.sort()
    confusion = metrics.confusion_matrix(y_train_pred, ground_truth)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.title(name + '_Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.grid(False)
    # plt.axis('off')
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            if first_index == 0 and second_index == 0:
                plt.text(first_index, second_index, confusion[first_index][second_index],color = 'w')
            else:
                plt.text(first_index, second_index, confusion[first_index][second_index])

    plt.savefig('matrix/' + name + '.png')




    plt.show()
    # 输出判断对哪些恶性、哪些良性、哪些无肿瘤
    

#     print("\ncorrect 恶性:(" + str(len(correct_tumor)) + ")", correct_tumor)

#     correct_tumor1 = []
#     for i in range(len(names)):
#         if ground_truth[i] == 1 and is_positive[i] == 0 and y_train_pred[i] == ground_truth[i]:
#             correct_tumor1.append(names[i])
#     print("\ncorrect 良性:(" + str(len(correct_tumor1)) + ")", correct_tumor1)

#     correct_no_tumor = []
#     for i in range(len(names)):
#         if ground_truth[i] == 0 and y_train_pred[i] == ground_truth[i]:
#             correct_no_tumor.append(names[i])
#     # print("\ncorrect no tumor:(" + str(len(correct_no_tumor)) + ")", correct_no_tumor[:85])






# # ground_truth = 15 * [1] + 1002 * [0]
# # y_train_pred = 14 * [1] + 1 * [0] + 124 * [1] + 878 * [0]
# #
# # result("1111",ground_truth=ground_truth,y_train_pred=y_train_pred)