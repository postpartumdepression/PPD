
######################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option('display.max_columns', None)#display all columns
from sklearn.preprocessing import StandardScaler

##############################################
###use delivery and stillbirth to define cohort 2014-2018
#delivery
delivery = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/delivery.csv', delimiter=',', header=0)
#print(delivery.head(2))
print(delivery.condition_start_date.min())#2004-08-30
print(delivery.condition_start_date.max())#2017-10-31
print(delivery.shape)#(132429, 45)
#drop duplicate
delivery = delivery.sort_values(['person_id','condition_start_date'], ascending=True)
delivery = delivery.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
print(delivery.shape)#wcm(56276, 28)#cdrn(99011, 45)

delivery['condition_start_date_lag'] = delivery.groupby(['person_id'])['condition_start_date'].transform(lambda x:x.shift())
delivery['diffday'] = pd.DataFrame(pd.to_datetime(delivery['condition_start_date_lag']) - pd.to_datetime(delivery['condition_start_date']))
delivery['days'] = delivery['diffday'].dt.days
type(delivery.days)

delivery_1 = delivery[delivery['days'].isnull()]
delivery_2 = delivery[delivery['days']>300]
delivery = pd.concat([delivery_1,delivery_2],axis = 0)
print(delivery.shape)#(24743, 31)#(71191, 48)

#stillbirth
stillbirth = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/stillbirth.csv', delimiter=',', header=0)
stillbirth.shape#(480, 28)
stillbirth = stillbirth.drop_duplicates(['person_id','condition_start_date'])
stillbirth.shape#(264, 28)
stillbirth = pd.DataFrame(stillbirth, columns=['person_id','condition_start_date'])
stillbirth['stillbirth'] = 1
stillbirth = stillbirth.rename({'condition_start_date': 'stillbirth_date'}, axis=1) #'a': 'X', 'b': 'Y'
#merge
delivery2 = pd.merge(delivery,stillbirth,on = ['person_id'], how='left')
#delivery2['stillbirth_lag'] = delivery2.groupby(['person_id'])['stillbirth_start_date'].transform(lambda x:x.shift())
delivery2['stillbirthday'] = pd.DataFrame(pd.to_datetime(delivery2['condition_start_date']) - pd.to_datetime(delivery2['stillbirth_date']))
delivery2['stillbirthday2'] = delivery2['stillbirthday'].dt.days
delivery2.shape#(24795, 35)
delivery2_1 = delivery2[delivery2['stillbirthday2']>180]#6 months
delivery2_2 = delivery2[delivery2['stillbirthday2']< (-20)]
delivery2_3 = delivery2[(delivery2['stillbirthday2'].isnull())] #| (df['b'].isnull()) 

delivery2 = pd.concat([delivery2_1,delivery2_2,delivery2_3],axis = 0)
delivery2 = delivery2.drop_duplicates(['person_id','condition_start_date'])#24623
delivery2 = delivery2.sort_values(['person_id','condition_start_date'], ascending=True)
#71105
##############################################
#miscarriage
miscarriage = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/miscarriage.csv', delimiter=',', header=0)
miscarriage.groupby('concept_name').count()
#drop duplicate
miscarriage = miscarriage.sort_values(['person_id','condition_start_date'], ascending=True)
miscarriage = miscarriage.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
miscarriage = pd.DataFrame(miscarriage, columns=['person_id','condition_start_date'])
miscarriage['miscarriage'] = 1
miscarriage = miscarriage.rename({'condition_start_date': 'miscarriage_date'}, axis=1) #'a': 'X', 'b': 'Y'

delivery2 = pd.merge(delivery2,miscarriage,on = ['person_id'], how='left')
delivery2['miscarriageday'] = pd.DataFrame(pd.to_datetime(delivery2['condition_start_date']) - pd.to_datetime(delivery2['miscarriage_date']))
delivery2['miscarriageday2'] = delivery2['miscarriageday'].dt.days
delivery2.shape#(32298, 39)
delivery2_1 = delivery2[delivery2['miscarriageday2']>180]
delivery2_2 = delivery2[delivery2['miscarriageday2']< (-20)]
delivery2_3 = delivery2[(delivery2['miscarriageday2'].isnull())] #| (df['b'].isnull()) 
delivery2 = pd.concat([delivery2_1,delivery2_2,delivery2_3],axis = 0)
delivery2 = delivery2.drop_duplicates(['person_id','condition_start_date'])#24594
delivery2 = delivery2.sort_values(['person_id','condition_start_date'], ascending=True)
#delivery2 is the total cohort (can/cannot calculate gestational week)
#71016
delivery2[['person_id']].groupby(delivery2['condition_type_concept_id']).agg(['count'])
#0                               557
#44786627                       3690
#44786629                      66165
#44814649                        272
#44814650                        332
#without the code 45905770 and 38000245

##############################################
#person
person = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/person.csv', delimiter=',', header=0)
#from datetime import datetime
#person['birthday'] = person.apply(lambda row: datetime(row['year_of_birth'], row['month_of_birth'], row['day_of_birth']), axis=1)
person = person.rename({'year_of_birth': 'Year', 'month_of_birth':'Month', 'day_of_birth':'Day'}, axis=1)
person['birthday'] = pd.to_datetime(person[['Year','Month','Day']])

#merge with deliverygw
deliverygw2 = pd.merge(delivery2,person,on = ['person_id'], how='left')
#calcualte age
deliverygw2['age'] = (pd.to_datetime(deliverygw2['condition_start_date']) - pd.to_datetime(deliverygw2['birthday']))/365.2425
deliverygw2['age'] = deliverygw2['age'].dt.days#71016
#limit age 18-45
deliverygw2 = deliverygw2[deliverygw2['age']>=18]
deliverygw2 = deliverygw2[deliverygw2['age']<=45]#69113

##############################################
#ppd -from 1w - 1y after childbirth
ppd = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/ppd.csv', delimiter=',', header=0)
ppd[['person_id']].groupby(ppd['concept_name']).agg(['count'])#'mean',
ppd = ppd.sort_values(['person_id','condition_start_date'], ascending=True)
ppd = ppd.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')

ppd = pd.DataFrame(ppd, columns=['person_id','condition_start_date','concept_name'])
ppd['ppd'] = 1
ppd = ppd.rename({'condition_start_date': 'ppd_date'}, axis=1)
###merge with deliverygw2
ppd = pd.merge(deliverygw2,ppd,on = ['person_id'], how='inner')
ppd['diffppd'] = pd.DataFrame(pd.to_datetime(ppd['ppd_date']) - pd.to_datetime(ppd['condition_start_date']))
ppd['diffppd'] = ppd['diffppd'].dt.days
ppd = ppd[(ppd['diffppd'] >= 0 ) & (ppd['diffppd'] < 367)]#0-367day 360;7-367days 189
ppd = ppd.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
ppd[['person_id']].groupby(ppd['concept_name']).agg(['count'])

#concept_name                                                
#Adjustment disorder with depressed mood                   76
#Anxiety disorder                                         677
#Generalized anxiety disorder                             128
#Major depression, single episode                          70
#Recurrent major depression                                 4
#Recurrent major depressive episodes, in full re...        29
#Severe recurrent major depression without psych...         2
#Single major depressive episode                          302
#Single major depressive episode, in full remission         6
#Single major depressive episode, mild                     24
#Single major depressive episode, moderate                 56

ppd = pd.DataFrame(ppd, columns=['person_id','condition_start_date','ppd_date','ppd','diffppd'])
deliverygw2 = pd.merge(deliverygw2,ppd,on = ['person_id','condition_start_date'], how='left')
deliverygw2['ppd']=deliverygw2['ppd'].fillna(0)
deliverygw2['ppd'].value_counts()
#0.0    67739
#1.0     1374

deliverygw2[['person_id']].groupby(deliverygw2['condition_type_concept_id']).agg(['count'])
#condition_type_concept_id          
#0                               546
#44786627                       3545
#44786629                      64440
#44814649                        267
#44814650                        315

##############################################
#antidepressants to define ppd
#include all codes under N06A, and then exclude the following, which are likely to have a dual indication for prescribing, in addition to mental health indication:
#N06AA09 Amitriptyline,Tryptizol
#N06AA04 Anafranil, Clomipramine
#N06AX21 Cymbalta,Duloxetine,Yentreve
#N05AF01 Fluanxol,Flupentixol
#N06AA10 Nortriptyline
#The reference for this approach is: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4828938/
medication_1 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/medication/medication_1.csv', delimiter=',', header=0)
medication_2 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/medication/medication_2.csv', delimiter=',', header=0)
medication = pd.concat([medication_1,medication_2],axis = 0)
#drug_type_concept_id IS NOT NULL and stop_reason is null and drug_exposure_end_date!=visit_start_date 

medication2 = medication[~pd.isnull(medication['drug_type_concept_id'])]
medication2 = medication2[pd.isnull(medication['stop_reason'])]


medication.head(2)
# String to be searched in start of string  
search ="N06A"
# series returned with False at place of NaN 
series = medication['concept_code'].str.startswith(search, na = False) 
# displaying filtered dataframe 
antidepressant = medication[series] #6195
#or
#antidepressant = medication[medication['concept_code.1'].str.contains('N06A')]
antidepressant['concept_class_id'].value_counts()
#ATC 5th    33958
#ATC 3rd    33958
#ATC 4th    33958
#then exclude codes before
antidepressant3 = antidepressant.loc[antidepressant['concept_class_id'].isin(['ATC 5th'])]
antidepressant3 = antidepressant3[~antidepressant3['concept_code'].isin(['N06AA09', 'N06AA04','N06AX21','N05AF01','N06AA10'])]#1851
antidepressant3['concept_name'].value_counts()  
#sertraline        8856
#escitalopram      4115
#bupropion         3844
#fluoxetine        3568
#citalopram        3250
#trazodone         2134
#venlafaxine       1445
#mirtazapine        992
#paroxetine         317
#doxepin            131
#desvenlafaxine     109
#fluvoxamine         92
#imipramine          25
#milnacipran         21
#protriptyline       17
#vortioxetine        17
#desipramine         13
#vilazodone          13
#nefazodone           5
#amoxapine            3
#Hyperici herba       1
#oxitriptan           1

antidepressant3 = pd.merge(deliverygw2,antidepressant3,on = ['person_id'], how='inner')
#calcualte age
antidepressant3['diffanti'] = pd.to_datetime(antidepressant3['drug_exposure_start_date']) - pd.to_datetime(antidepressant3['condition_start_date'])
antidepressant3['diffanti'] = antidepressant3['diffanti'].dt.days
antidepressant3 = antidepressant3[(antidepressant3['diffanti'] >= 0 ) & (antidepressant3['diffanti'] < 367)]
antidepressant3 = antidepressant3.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
antidepressant3['ppd'].value_counts()
antidepressant3 = pd.DataFrame(antidepressant3, columns=['person_id','condition_start_date','drug_exposure_start_date','diffanti','ppd'])
antidepressant3 = antidepressant3[(antidepressant3['ppd'] == 0 )]
antidepressant3 = antidepressant3.drop('ppd', 1)#0 for rows and 1 for columns
#merge deliverygw2
#diffanti, drug_exposure_start_date
#diffppd, ppd_date
deliverygw3 = pd.merge(deliverygw2,antidepressant3,on = ['person_id','condition_start_date'], how='left')
deliverygw3.loc[deliverygw3['diffppd'].isnull(),'diffppd']=deliverygw3[deliverygw3['diffppd'].isnull()]['diffanti']
deliverygw3.loc[deliverygw3['ppd_date'].isnull(),'ppd_date']=deliverygw3[deliverygw3['ppd_date'].isnull()]['drug_exposure_start_date']
deliverygw3.ppd[deliverygw3.diffppd > 0] = 1
deliverygw3 = deliverygw3.drop('drug_exposure_start_date', 1)
deliverygw3['ppd'].value_counts()
#0.0    66818
#1.0     2295

##############################################
#condition
#deliverygw4['gestationday'] = deliverygw4['fmyz'] * 7
deliverygw4 = deliverygw3

condition_1 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/condition/condition_1.txt', delimiter=',', header=0)
condition_2 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/condition/condition_2.txt', delimiter=',', header=0)
condition = pd.concat([condition_1,condition_2],axis = 0)
#condition['person_id'].groupby(condition['condition_type_concept_id']).agg(['count'])
#condition_type_concept_id         
#0                           199033
#44786627                    855742
#44786629                   1670180
#44814649                     30551
#44814650                   5532949

condition = condition.sort_values(['person_id','condition_start_date','concept_name'], ascending=True)
condition = condition.drop_duplicates(subset=['person_id','condition_start_date','concept_name'], keep='first')
condition = condition.rename({'condition_start_date': 'condition_date'}, axis=1) #'a': 'X', 'b': 'Y'
condition = pd.merge(deliverygw4,condition,on = ['person_id'], how='inner')
condition['diffcondition'] = pd.to_datetime(condition['condition_start_date']) - pd.to_datetime(condition['condition_date'])
condition['diffcondition'] = condition['diffcondition'].dt.days

#condition = condition[(condition['diffcondition'] >= 0 ) & (condition['diffcondition'] <= 210)] #30*7
condition = condition[(280 - condition['diffcondition']) <= 210 ]
condition = condition[(280 - condition['diffcondition']) >= 0 ]

condition = condition.drop_duplicates(subset=['person_id','condition_start_date','concept_name'], keep='first')
condition = pd.DataFrame(condition, columns=['person_id','condition_start_date','condition_date','concept_name','diffcondition'])

#rename the new columns for condition name
condition_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/condition_listcombine.csv', delimiter=',', header=0)
condition_list = condition_list.rename({'concept_name_y': 'concept_name'}, axis=1) #'a': 'X', 'b': 'Y'

#!condition_list
condition = pd.merge(condition,condition_list,on = ['concept_name'], how='left')
condition.isnull().sum(axis=0)
#null_data = condition[condition.isnull().any(axis=1)]#output null data
condition2 = pd.DataFrame(condition, columns=['person_id','condition_start_date','newvariable'])
condition2 = condition2.drop_duplicates(subset=['person_id','condition_start_date','newvariable'], keep='first')
conditioncount = condition2.groupby('newvariable').count()


condition2['values'] = 1
condition2 = (condition2.pivot_table(index=['person_id','condition_start_date'], columns='newvariable',values='values').reset_index())



##############################################
#medication
medication_1 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/medication/medication_1.csv', delimiter=',', header=0)
medication_2 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/medication/medication_2.csv', delimiter=',', header=0)
medication = pd.concat([medication_1,medication_2],axis = 0)

medication2 = medication.loc[medication['concept_class_id'].isin(['ATC 3rd'])]
medication2 = medication2.sort_values(['person_id','drug_exposure_start_date','concept_name'], ascending=True)
medication2 = medication2.drop_duplicates(subset=['person_id','drug_exposure_start_date','concept_name'], keep='first')
medication2 = pd.DataFrame(medication2, columns=['person_id','drug_exposure_start_date','concept_name'])

medication2 = pd.merge(deliverygw4,medication2,on = ['person_id'], how='inner')
medication2['diffmedication'] = pd.to_datetime(medication2['condition_start_date']) - pd.to_datetime(medication2['drug_exposure_start_date'])
medication2['diffmedication'] = medication2['diffmedication'].dt.days

medication2 = medication2[(280 - medication2['diffmedication']) <= 210 ]
medication2 = medication2[(280 - medication2['diffmedication']) >= 0 ]

medication2 = medication2.drop_duplicates(subset=['person_id','drug_exposure_start_date','concept_name'], keep='first')
medication2 = pd.DataFrame(medication2, columns=['person_id','drug_exposure_start_date','condition_start_date','concept_name','diffmedication'])

#rename the new columns for condition name
medication_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/medication_listnew.csv', delimiter=',', header=0)
medication_list = medication_list.rename({'concept_name.1': 'concept_name'}, axis=1) #'a': 'X', 'b': 'Y'

#!medication_list
medication2 = pd.merge(medication2,medication_list,on = ['concept_name'], how='left')
medication2.isnull().sum(axis=0)


medication2 = pd.DataFrame(medication2, columns=['person_id','condition_start_date','newvariable'])
medication2 = medication2.drop_duplicates(subset=['person_id','condition_start_date','newvariable'], keep='first')
#remove frequency below 10
#v = medication2[['newvariable']]
#medication2 = medication2[v.replace(v.apply(pd.Series.value_counts)).gt(10).all(1)]
#medication2.groupby('newvariable').count()
#pivot_table() to transform the data (same with cast/dcast in R)
medication2['values'] = 1
medication2 = (medication2.pivot_table(index=['person_id','condition_start_date'], columns='newvariable',values='values').reset_index())


##############################################
#pre-pregnancy BMI
bmi_1 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/measurement/bmi_1.csv', delimiter=',', header=0)
bmi_2 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/measurement/bmi_2.csv', delimiter=',', header=0)
bmi = pd.concat([bmi_1,bmi_2],axis = 0)
bmi = bmi[(bmi['value_as_number'] >10 ) & (bmi['value_as_number'] < 60)]

bmi = pd.DataFrame(bmi, columns=['person_id','measurement_date','value_as_number'])
bmi = pd.merge(deliverygw4,bmi,on = ['person_id'], how='inner')
bmi['diffbmi'] = pd.to_datetime(bmi['condition_start_date']) - pd.to_datetime(bmi['measurement_date'])
bmi['diffbmi'] = bmi['diffbmi'].dt.days - 280
bmi = bmi.sort_values(['person_id','condition_start_date','measurement_date'], ascending=True)
bmi['diffbmi'] = np.abs(bmi['diffbmi'])
bmi = bmi.sort_values(['person_id','condition_start_date','diffbmi']).groupby(['person_id','condition_start_date'], as_index=False).first()
bmi = pd.DataFrame(bmi, columns=['person_id','condition_start_date','value_as_number'])
bmi = bmi.rename({'value_as_number': 'bmi'}, axis=1) 


##############################################
#bp
bp = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/measurement/bpcdrn.csv', delimiter=',', header=0)
plt.hist(bp['value_as_number'], bins=50, color='steelblue', normed=True)
bp = bp[(bp['value_as_number'] >30 )]
bp = bp[(bp['value_as_number'] <200 )]

bp = pd.DataFrame(bp, columns=['person_id','measurement_date','value_as_number','concept_name'])
bp = pd.merge(deliverygw4,bp,on = ['person_id'], how='inner')
bp['diffbp'] = pd.to_datetime(bp['condition_start_date']) - pd.to_datetime(bp['measurement_date'])
bp['diffbp'] = bp['diffbp'].dt.days
bp = bp[(bp['diffbp'] >= 0 ) & (bp['diffbp'] <= 280)]

def trimester(bp):
    if ((280 - bp['diffbp']) < 84):
        return 1
    if ((280 - bp['diffbp']) >=84) and ((280 - bp['diffbp']) < 196):
        return 2
    else:
        return 3
bp['trim'] = bp.apply(trimester, axis=1)
bp.groupby('concept_name').count()

sbp_1 = bp[(bp['concept_name'] == 'Systolic blood pressure--sitting')]
sbp_2 = bp[(bp['concept_name'] == 'Systolic blood pressure--standing')]
sbp_3 = bp[(bp['concept_name'] == 'Systolic blood pressure--supine')]

dbp_1 = bp[(bp['concept_name'] == 'Systolic blood pressure--sitting')]
dbp_2 = bp[(bp['concept_name'] == 'Systolic blood pressure--standing')]
dbp_3 = bp[(bp['concept_name'] == 'Systolic blood pressure--supine')]

sbp = pd.concat([sbp_1,sbp_2,sbp_3],axis = 0)
dbp = pd.concat([dbp_1,dbp_2,dbp_3],axis = 0)

sbp['sbpmean'] = sbp.groupby(['person_id','condition_start_date','trim'])['value_as_number'].transform('mean')
sbp = pd.DataFrame(sbp, columns=['person_id','condition_start_date','sbpmean','trim'])
sbp = sbp.drop_duplicates(subset=['person_id','condition_start_date','sbpmean','trim'], keep='first')
dbp['dbpmean'] = dbp.groupby(['person_id','condition_start_date','trim'])['value_as_number'].transform('mean')
dbp = pd.DataFrame(dbp, columns=['person_id','condition_start_date','dbpmean','trim'])
dbp = dbp.drop_duplicates(subset=['person_id','condition_start_date','dbpmean','trim'], keep='first')

sbp = (sbp.pivot_table(index=['person_id','condition_start_date'], columns='trim',values='sbpmean').reset_index())
sbp = sbp.rename({1: 'sbp1st',2: 'sbp2nd',3: 'sbp3rd'}, axis=1) #'a': 'X', 'b': 'Y'
dbp = (dbp.pivot_table(index=['person_id','condition_start_date'], columns='trim',values='dbpmean').reset_index())
dbp = dbp.rename({1: 'dbp1st',2: 'dbp2nd',3: 'dbp3rd'}, axis=1) #'a': 'X', 'b': 'Y'

bp = pd.merge(sbp,dbp,on = ['person_id','condition_start_date'], how='outer')


##############################################
#weight
weight_1 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/measurement/weight_1.txt', delimiter=',', header=0)
weight_2 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/measurement/weight_2.txt', delimiter=',', header=0)
weight = pd.concat([weight_1,weight_2],axis = 0)

#Remove strings from a float number in a column
#weight['weight2'] = pd.to_numeric(weight.value_source_value.str.replace('[^d.]', ''), errors='coerce')
#plt.hist(weight['weight2'], bins=50, color='steelblue', normed=True)
weight = weight[(weight['value_as_number'] >20 ) & (weight['value_as_number'] < 300)]

#plt.hist(weight['weight2'], bins=50, color='steelblue', normed=True)
weight = pd.DataFrame(weight, columns=['person_id','measurement_date','value_as_number'])
weight = pd.merge(deliverygw4,weight,on = ['person_id'], how='inner')
weight['diffweight'] = pd.to_datetime(weight['condition_start_date']) - pd.to_datetime(weight['measurement_date'])
weight['diffweight'] = weight['diffweight'].dt.days
weight = weight[(weight['diffweight'] >= 0 ) & (weight['diffweight'] <= 280)]

def trimester(weight):
    if ((280 - weight['diffweight']) < 84):
        return 1
    if ((280 - weight['diffweight']) >=84) and ((280 - weight['diffweight']) < 196):
        return 2
    else:
        return 3
weight['trim'] = weight.apply(trimester, axis=1)

weight['weightmean'] = weight.groupby(['person_id','condition_start_date','trim'])['value_as_number'].transform('mean')
weight = pd.DataFrame(weight, columns=['person_id','condition_start_date','weightmean','trim'])
weight = weight.drop_duplicates(subset=['person_id','condition_start_date','weightmean','trim'], keep='first')

weight = (weight.pivot_table(index=['person_id','condition_start_date'], columns='trim',values='weightmean').reset_index())
weight = weight.rename({1: 'weight1st',2: 'weight2nd',3: 'weight3rd'}, axis=1) #'a': 'X', 'b': 'Y'

##############################################
#amniocentesis
amniocentesis_1 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/amniocentesis_1.csv', delimiter=',', header=0)
amniocentesis_2 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/amniocentesis_2.csv', delimiter=',', header=0)
amniocentesis = pd.concat([amniocentesis_1,amniocentesis_2],axis = 0)

amniocentesis.groupby(['concept_name','concept_id']).count()
amniocentesis = amniocentesis[~amniocentesis['concept_id'].isin(['2110331','2110332'])]
amniocentesis = pd.DataFrame(amniocentesis, columns=['person_id','procedure_date'])
amniocentesis = pd.merge(deliverygw4,amniocentesis,on = ['person_id'], how='inner')
amniocentesis['diffamniocentesis'] = pd.to_datetime(amniocentesis['condition_start_date']) - pd.to_datetime(amniocentesis['procedure_date'])
amniocentesis['diffamniocentesis'] = amniocentesis['diffamniocentesis'].dt.days

amniocentesis = amniocentesis[(amniocentesis['diffamniocentesis'] >= 0 ) & (amniocentesis['diffamniocentesis'] <= 280)]
amniocentesis = amniocentesis.drop_duplicates(subset=['person_id','procedure_date'], keep='first')
amniocentesis = pd.DataFrame(amniocentesis, columns=['person_id','condition_start_date'])
amniocentesis['amniocentesis'] = 1
amniocentesis = amniocentesis.drop_duplicates()


##############################################
#visitoccurrence edvisit
visit_1 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/visit/visit_1.csv', delimiter=',', header=0)
visit_2 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/visit/visit_2.csv', delimiter=',', header=0)
visit = pd.concat([visit_1,visit_2],axis = 0)

visit = visit[visit['visit_concept_id'].isin(['262','9203'])]
visit = visit.drop_duplicates(subset=['person_id','visit_start_date'], keep='first')

visit = pd.merge(deliverygw4,visit,on = ['person_id'], how='inner')
visit['diffvisit'] = pd.to_datetime(visit['condition_start_date']) - pd.to_datetime(visit['visit_start_date_x'])
visit['diffvisit'] = visit['diffvisit'].dt.days
visit = visit[(visit['diffvisit'] >= 0 ) & (visit['diffvisit'] <= 210)]

visit['value'] = 1
visit = visit.groupby(['person_id','condition_start_date'])['value'].count().reset_index(name="edvisitcount")

##############################################
#merge data to build the model file
def race(a):
    if (a['race_concept_id_x'] == 8527) or (a['race_concept_id_x'] == 8557):
        return 1
    if (a['race_concept_id_x'] == 8515):#aisa
        return 2
    if (a['race_concept_id_x'] == 8657) or (a['race_concept_id_x'] == 8516):#Black or African American
        return 3
    if (a['race_concept_id_x'] == 44814659) or (a['race_concept_id_x'] == 44814649):
        return 4#Other
    else:
        return 5#Unknown

#person4<-within(person4,{
#  racegroup<-NA
#  racegroup[race_name=='White']<-1 8527
#  racegroup[race_name=='Asian']<-2 8515
#  racegroup[race_name=='American Indian or Alaska Native']<-3 8657 
#  racegroup[race_name=='Black or African American']<-4  8516
#  racegroup[race_name=='Native Hawaiian or Other Pacific Islander']<-5 8557
#  racegroup[race_name=='No information'|race_name=='Refuse to answer'|race_name=='Unknown']<-6 44814650 44814660 44814660
#  racegroup[race_name=='Multiple race'|race_name=='Other']<-7  44814659 44814649
#})   
    
    
    
deliverygw4['race2'] = deliverygw4.apply(race, axis=1)
demo = pd.DataFrame(deliverygw4, columns=['person_id','condition_start_date','race2','age','ppd'])


#the fequency of 'Studenthealthplan','INTERNATIONAL' are below 10
#insurance = insurance.drop(['Studenthealthplan','INTERNATIONAL'], axis=1)

#demo
#condition2
#medication2
#bp
#weight
#insurance 
#amniocentesis
#visit
#marital
#bmi

#wcmmodel = pd.merge(demo,marital,on = ['person_id','condition_start_date'], how='left')
#wcmmodel['marital']=wcmmodel['marital'].fillna(5)#5-not known
cdrnmodel = pd.merge(demo,visit,on = ['person_id','condition_start_date'], how='left')
cdrnmodel['edvisitcount']=cdrnmodel['edvisitcount'].fillna(0)
cdrnmodel = pd.merge(cdrnmodel,amniocentesis,on = ['person_id','condition_start_date'], how='left')
cdrnmodel['amniocentesis']=cdrnmodel['amniocentesis'].fillna(0)

#cdrnmodel = pd.merge(cdrnmodel,insurance,on = ['person_id','condition_start_date'], how='left')
#cdrnmodel=cdrnmodel.fillna(0)
cdrnmodel = pd.merge(cdrnmodel,weight,on = ['person_id','condition_start_date'], how='left')
cdrnmodel = pd.merge(cdrnmodel,bp,on = ['person_id','condition_start_date'], how='left')
cdrnmodel = pd.merge(cdrnmodel,bmi,on = ['person_id','condition_start_date'], how='left')
cdrnmodel = cdrnmodel.fillna(cdrnmodel.mean())

one_hotrace = pd.get_dummies(cdrnmodel['race2'])
one_hotrace = one_hotrace.rename({1: 'white',2:'asian',3:'black',4:'other',5:'unknown'}, axis=1) #'a': 'X', 'b': 'Y'
cdrnmodel = cdrnmodel.drop('race2',axis = 1)
cdrnmodel = cdrnmodel.join(one_hotrace)

#Normalize data
#normalize = pd.DataFrame(cdrnmodel, columns=['fmyz','age','weight1st','weight2nd','weight3rd','sbp1st','sbp2nd','sbp3rd','dbp1st','dbp2nd','dbp3rd'])
num_cols = cdrnmodel[['age','weight1st','weight2nd','weight3rd','sbp1st','sbp2nd','sbp3rd','dbp1st','dbp2nd','dbp3rd','bmi']]
scaler = StandardScaler()
num_cols = scaler.fit_transform(num_cols)
normalize = pd.DataFrame(num_cols)
normalize.rename(columns={
                          1:'weight1st',
                          2:'weight2nd',
                          0:'age',
                          3:'weight3rd',
                          4:'sbp1st',
                          5:'sbp2nd',
                          6:'sbp3rd',
                          7:'dbp1st',
                          8:'dbp2nd',
                          9:'dbp3rd',
                          10:'bmi'
                          }, 
                 inplace=True)
cdrnmodel['weight1st'] = normalize['weight1st']
cdrnmodel['weight2nd'] = normalize['weight2nd']
cdrnmodel['age'] = normalize['age']
cdrnmodel['weight3rd'] = normalize['weight3rd']
cdrnmodel['sbp1st'] = normalize['sbp1st']
cdrnmodel['sbp2nd'] = normalize['sbp2nd']
cdrnmodel['sbp3rd'] = normalize['sbp3rd']
cdrnmodel['dbp1st'] = normalize['dbp1st']
cdrnmodel['dbp2nd'] = normalize['dbp2nd']
cdrnmodel['dbp3rd'] = normalize['dbp3rd']
cdrnmodel['bmi'] = normalize['bmi']
cdrnmodel = cdrnmodel.drop(['weight3rd','sbp3rd','dbp3rd'], axis=1)

#merge with diagnose/medication
cdrnmodel = pd.merge(cdrnmodel,condition2,on = ['person_id','condition_start_date'], how='left')
cdrnmodel = pd.merge(cdrnmodel,medication2,on = ['person_id','condition_start_date'], how='left')
cdrnmodel=cdrnmodel.fillna(0)
cdrnmodel['ppd'].value_counts()
#cdrnmodel.to_csv('/Users/bu/Desktop/CDRN-data/cdrnmodelfile/cdrnmodel.csv', sep=',',index=0)

##############################################
#history

def func(deliverygw4, disease="anxiety", folder="/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/cdrn/history/"):
    file_dir = folder+disease+".csv"
    data = pd.read_csv(file_dir, delimiter=',', header=0)
    data = data.sort_values(['person_id','condition_start_date'], ascending=True)
    data = data.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
    data[disease] = 1
    data = data.rename({'condition_start_date': disease+"_date"}, axis=1) #'a': 'X', 'b': 'Y'
    data = pd.merge(deliverygw4,data,on = ['person_id'], how='inner')
    data["diff"+disease] = pd.DataFrame(pd.to_datetime(data[disease+"_date"]) - pd.to_datetime(data['condition_start_date']))
    data["diff"+disease] = data["diff"+disease].dt.days
    data = data[(data["diff"+disease] > 0 )]
    data = data.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
    data = pd.DataFrame(data, columns=['person_id','condition_start_date',disease])
    return data

def main():
    global new_deliverygw4
    #deliverygw3 = pd.read_csv('/Users/shuojiawang/Documents/ppdmodel/orifile/delivery.csv', delimiter=',', header=0)
    #print(deliverygw3.head())
        
    disease_list = ["anxiety", "mooddisorder", "organicdisorders", "otherdisorder", "personalitydisorder", "schizophrenic", "substance"]
    #disease_list = ["anxiety", "mooddisorder"]
    
    new_deliverygw4 = deliverygw4
    for d in disease_list:
        disease_table = func(deliverygw4, disease=d)
        new_deliverygw4 = pd.merge(new_deliverygw4,disease_table,on = ['person_id','condition_start_date'], how='left')
#        print(disease_table)
    #print(new_deliverygw4)

if __name__ == "__main__":
    main()


history = pd.DataFrame(new_deliverygw4, columns=['person_id','condition_start_date',"anxiety", "mooddisorder", "organicdisorders", "otherdisorder", "personalitydisorder", "schizophrenic", "substance"])
history = history.fillna(0)
cdrnmodelhistory = pd.merge(cdrnmodel,history,on = ['person_id','condition_start_date'], how='left')
# cdrnmodel=cdrnmodel.fillna(0)
# cdrnmodel['ppd'].value_counts()
cdrnmodelhistory.to_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30_2.csv', sep=',',index=0)












