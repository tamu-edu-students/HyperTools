import json
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # load lib hsi json file that has the class names
    f = open('../json/lib_hsi.json')
    
    # location of results json files
    results_dir = '../../HyperImages/LIB-HSI/LIB-HSI/validation/results/'   
    
    # number of spectral similarity algorithms
    # num_algorithms = 14 # includes zero,  14 for hypertools 
    
    # to do incorporate time / number of classes? 
    
    
    ## get results from json file
    
    files_list = os.listdir(results_dir)
    results_dict_alg = {}
    results_dict_class = {}
    temp_counter=0
    for file in files_list:
        # print (file)
        f = open(results_dir + file)
        try: 
            
            jsonAsDict = json.load(f)
            ss_results = jsonAsDict['Spectral Similarity Algorithm']
            temp_counter+=1
            # print(ss_results)
            for alg_num in ss_results:
                # create dictionary for each algorithm, if it does not exist
                # print(alg_num)
                if alg_num not in results_dict_alg:
                    results_dict_alg[alg_num]={}
                    for key in ss_results[alg_num]:
                        # print(key)
                        last_five_chars = key[-5:]
                        # print(last_five_chars)
                        if last_five_chars != 'Class':
                            results_dict_alg[alg_num][key] = ss_results[alg_num][key]
                        else:
                            for class_name in ss_results[alg_num][key]:

                                if alg_num not in results_dict_class:
                                    results_dict_class[alg_num] = {}

                                if class_name not in results_dict_class[alg_num]:
                                    results_dict_class[alg_num][class_name] = {}

                                classification_key = key[:-6]
                                if classification_key not in results_dict_class[alg_num][class_name]:

                                    results_dict_class[alg_num][class_name][classification_key] = ss_results[alg_num][key][class_name]
                                else:
                                    results_dict_class[alg_num][class_name][classification_key] += ss_results[alg_num][key][class_name]

                else :
                    for key in ss_results[alg_num]:
                        last_five_chars = key[-5:]
                        if last_five_chars != 'Class':
                            results_dict_alg[alg_num][key] += ss_results[alg_num][key]
                        else:
                            for class_name in ss_results[alg_num][key]:

                                if alg_num not in results_dict_class:
                                    results_dict_class[alg_num] = {}

                                if class_name not in results_dict_class[alg_num]:
                                    results_dict_class[alg_num][class_name] = {}

                                classification_key = key[:-6]
                                if classification_key not in results_dict_class[alg_num][class_name]:

                                    results_dict_class[alg_num][class_name][classification_key] = ss_results[alg_num][key][class_name]
                                else:
                                    results_dict_class[alg_num][class_name][classification_key] += ss_results[alg_num][key][class_name]
                    
                # for alg_results in ss_results[alg_num]:
                #     try:
                #         results_dict_alg[alg_num][alg_results] += ss_results[alg_num][alg_results]
                #     except:
                #         print('Error: ', file)
                #         results_dict_alg[alg_num][alg_results] = ss_results[alg_num][alg_results]
                        
                    # for alg_results in ss_results[alg_num]:
                    # # print(alg_results)
                    # results_dict_alg[alg_num][alg_results] = ss_results[alg_num][alg_results]
            
                    
                #     # print(alg_num)
                #     for alg_results in ss_results[alg_num]:
                #         # print(alg_results)
                #         results_dict_alg[alg_num][alg_results] = ss_results[alg_num][alg_results]
                # else:
                #     for alg_results in ss_results[alg_num]:
                #         results_dict_alg[alg_num][alg_results] += ss_results[alg_num][alg_results]
            # print('Attempting: ', file)
        except:
            print('Error: ', file)
            pass
           
    # print(temp_counter)
    # print(results_dict_alg)
    # import sys
    # sys.exit()
    
    # print the results
    # print ('Algorithm, Class Name, Correct, Incorrect, Total')
    
    recallDictAlg = {}
    precisionDictAlg = {}
    f1DictAlg = {}
    miouDictAlg = {}
    for item in results_dict_alg:

        FP = results_dict_alg[item]['False Positive']
        TP = results_dict_alg[item]['True Positive']
        TN = results_dict_alg[item]['True Negative']
        FN = results_dict_alg[item]['False Negative']
        NumClasses = results_dict_alg[item]['Number of Classes']
        NumPixels = results_dict_alg[item]['Number of Pixels']
        Time = results_dict_alg[item]['Time']

        #tp/numpixels weighted avg for total num that are correct
        #avg accuracy for one with TN and one without 
   
        # print ('Algorithm Number: ', item   ,  results_dict_alg[item]) # alg number
        print ('Algorithm Number:', item   ,  ' Total Accuracy:', TP/NumPixels*100,'% ', 'Total Inaccuracy:', FP/NumPixels*100,'% ', 'Classification Time per Class (s):', NumClasses/Time ) # alg number
        # print ('\nAlgorithm Number:', item, ' Total Accuracy: {:.3f}%'.format(TP/NumPixels*100), 'Total Inaccuracy: {:.3f}%'.format(FP/NumPixels*100), 'Classification Time per Class (s): {:.3f}'.format(NumClasses/Time), 'Number of Images:', len(files_list)) # alg number
        # print('Algorithm: ',item, ', Correct: ', alg_correct,' ',  alg_correct/alg_total*100,'% , Incorrect:', alg_incorrect, ' ', alg_incorrect/alg_total*100,'% , Total:', alg_total)

        #calculating miou for each class


        print("Metrics")
        # recall calculation
        recall = TP / (TP + FN) 
        # precision calculation
        precision = TP / (TP + FP)
        # f1 calculation
        f1 = 2 * (precision * recall) / (precision + recall)
    

        #print("Class Metrics")
        precisionSum = []
        recallSum = []
        f1Sum = []
        mIoUSum = []       

        for class_name in results_dict_class[item]:
            class_results = results_dict_class[item][class_name]
            TP = class_results['True Positive']
            FP = class_results['False Positive']
            FN = class_results['False Negative']
            TN = class_results['True Negative']
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)
            # -- These Print statements are for each class per algorithm --
            #print('Class:', class_name)
            #print(TP, FP, FN, TN)
            #print('Recall:', recall)
            recallSum.append(recall)
            #print('Precision:', precision)
            precisionSum.append(precision)
            #print('F1 Score:', f1)
            f1Sum.append(f1)
            #print()

    
            mIoUSum.append(TP / (TP+ FP + FN))

        
        recallDictAlg[item] = sum(recallSum) / len(recallSum)
        precisionDictAlg[item] = sum(precisionSum) / len(precisionSum)
        f1DictAlg[item] = sum(f1Sum) / len(f1Sum)
        miouDictAlg[item] = sum(mIoUSum) / len(mIoUSum)

        print("Average Recall Metrics for Algorithm",item," ", sum(recallSum) / len(recallSum))
        print("Average Precision Metrics for Algorithm", item," ",sum(precisionSum) / len(precisionSum))
        print("Average F1 Metrics for Algorithm",item," ", sum(f1Sum) / len(f1Sum))
        print("mIoU:", sum(mIoUSum) / len(mIoUSum))
        print()

    keys = f1DictAlg.keys()
    f1_values = f1DictAlg.values()
    recall_values = recallDictAlg.values()
    precision_values = precisionDictAlg.values()
    mIou_values = miouDictAlg.values()

    barWidth = 0.25
    r1 = np.arange(len(f1_values))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    plt.bar(r1, f1_values, color='b', width=barWidth, edgecolor='grey', label='F1 Score')
    plt.bar(r2, recall_values, color='g', width=barWidth, edgecolor='grey', label='Recall')
    plt.bar(r3, precision_values, color='r', width=barWidth, edgecolor='grey', label='Precision')
    plt.bar(r4, mIou_values, color='y', width=barWidth, edgecolor='grey', label='mIoU')

    plt.title('Metrics by Algorithm')
    plt.xlabel('Algorithm Number')
    plt.ylabel('Metrics')
    plt.xticks([r + barWidth for r in range(len(f1_values))], keys)

    plt.legend(fontsize='xx-small', loc='lower right')

    plt.show()

