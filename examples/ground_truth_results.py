import json
import os


if __name__ == "__main__":

    # load lib hsi json file that has the class names
    f = open('json/lib_hsi.json')
    
    # location of results json files
    results_dir = '../HyperImages/LIB-HSI/LIB-HSI/validation/results/'   
    
    # number of spectral similarity algorithms
    num_algorithms = 15 # includes zero,  15 for hypertools 
    
    # to do incorporate time / number of classes? 
    
    
    # get list of class names
    jsonAsDict = json.load(f)
    ClassNames =  jsonAsDict['items']     
    Class_List = [] # list of class names
        
    for item in ClassNames:
        # print(item['name'])
        Class_List.append(item['name'])
    
    # print(Class_List)
    
    # initialize dictionary with class names, number of correct classifications, number of incorrect classifications, and total number of classifications by algorithm for each class
    results_dict_alg = {}
    
    for item in range(num_algorithms+1):
        # print(item)
        results_dict_alg[item]={}
        results_dict_alg[item].update({'algorithm':item})  # algorithm number in hypertools, can be changed to algorithm name
        
        # for each algorithm, create a dictionary with the class names
        results_dict_alg[item]['algorithm_results']={}
        
        for name in Class_List:
            results_dict_alg[item]['algorithm_results'].update({name:{'correct_classification':0, 'incorrect_classification':0, 'total_classification':0}})
            # number of correct classification, number of incorrect classification, total number of of pixels in ground truth images
        
    # print (results_dict_alg)

    # load name of files in results directory
    files_list = os.listdir(results_dir)
    # print(files_list)
    for file in files_list:
        # print(file)
        # load results json file
        f = open(results_dir + file)
        jsonAsDict = json.load(f)
        # print(jsonAsDict)
        ss_results = jsonAsDict['Spectral Similarity Algorithm']
        num_pixels = jsonAsDict['Image_info']
        
        # add total num pixels to dictionary
        for name in num_pixels:
            # print(name)
            for item in range(num_algorithms+1):
                results_dict_alg[item]['algorithm_results'][name]['total_classification'] += num_pixels[name]['num_pixels']
                 
        
        # if len(ss_results) > num_algorithms:
        #     print('Error: number of algorithms does not match number of results')
        #     print('Number of algorithms should be at least: ', len(ss_results))
        #     break
        
        # add correct and incorrect classifications to dictionary
        for alg_num in ss_results:
            # print(alg_num)
            # # print(ss_results[name])
            for item in range(num_algorithms+1):
                for key in ss_results[alg_num]:
                    # print(key) # class name
                    if key == 'Number of Classes' or key == 'Time':
                        continue
                    # print(results_dict_alg[int(alg_num)])
                    results_dict_alg[int(alg_num)]['algorithm_results'][key]['correct_classification'] += ss_results[(alg_num)][key]['correct']
                    results_dict_alg[int(alg_num)]['algorithm_results'][key]['incorrect_classification'] += ss_results[(alg_num)][key]['incorrect']
                    # results_dict_alg[item]['algorithm_results'][name]['incorrect_classification'] += 1
                    pass
        #     break # uncommment to only do for one algorithm
        
        # break # uncommment to only do for one file
        
        # for name in Class_List:
        #     try:
        #         print(num_pixels[name]['num_pixels'])
        #     except:
        #         pass
        
        # break
    # print size of ss_results
    # print (len(ss_results))

    
    # print (results_dict_alg)
    # print first column of class names, then correct, incorrect, total
    # print ('Algorithm, Class Name, Correct, Incorrect, Total')
    for item in results_dict_alg:
        # print (item) # alg number
        pass
        for name in results_dict_alg[item]['algorithm_results']:
            # print (name) # class name
            # print (results_dict_alg[item]['algorithm_results'][name]['correct_classification'])
            # print (results_dict_alg[item]['algorithm_results'][name]['incorrect_classification'])
            # print (results_dict_alg[item]['algorithm_results'][name]['total_classification'])
            print ('Algorithm: ',item, ',', name, ', Correct: ', results_dict_alg[item]['algorithm_results'][name]['correct_classification'], ', Incorrect:', results_dict_alg[item]['algorithm_results'][name]['incorrect_classification'], ', Total:', results_dict_alg[item]['algorithm_results'][name]['total_classification'])
            pass
    
    
    
    
    
    # print (type(ss_results))
    # for x in ss_results:
    #     # print(ss_results[x])
    #     print (type(ss_results[x]))
    #     for y in ss_results[x]:
    #         print(y)
        
    
    
    
    
    
    
    