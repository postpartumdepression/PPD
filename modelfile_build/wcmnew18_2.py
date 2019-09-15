#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 08:21:40 2019

@author: bu
"""

######################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option('display.max_columns', None)#display all columns
from sklearn.preprocessing import StandardScaler
folder="/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/wcm/"
##############################################
###use delivery and stillbirth to define cohort 2014-2018
#delivery
delivery = pd.read_csv(folder+ "delivery.csv", delimiter=',', header=0)
print(delivery.shape)#(214359, 28)
#onlyinclude delivery date between 20141231-20180631
mask = (delivery['condition_start_date'] > '2014-12-31') & (delivery['condition_start_date'] < '2018-07-01' ) 
delivery = delivery.loc[mask]
print(delivery.condition_start_date.min())
print(delivery.condition_start_date.max())
print(delivery.shape)#(170527, 28)
#drop duplicate
delivery = delivery.sort_values(['person_id','condition_start_date'], ascending=True)
delivery = delivery.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
print(delivery.shape)#(56276, 28)
#distinguish duplicate deliveries records
delivery['condition_start_date_lag'] = delivery.groupby(['person_id'])['condition_start_date'].transform(lambda x:x.shift())
delivery['diffday'] = pd.DataFrame(pd.to_datetime(delivery['condition_start_date_lag']) - pd.to_datetime(delivery['condition_start_date']))
delivery['days'] = delivery['diffday'].dt.days

delivery_1 = delivery[delivery['days'].isnull()]
delivery_2 = delivery[delivery['days']>300]
delivery = pd.concat([delivery_1,delivery_2],axis = 0)
print(delivery.shape)#(24743, 31)
##############################################
#exclude stillbirth
stillbirth = pd.read_csv(folder+ "stillbirth.csv", delimiter=',', header=0)
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

##############################################
#exclude miscarriage
miscarriage = pd.read_csv(folder+ "miscarriage.csv", delimiter=',', header=0)
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

##############################################
#gestational week
gestationalweek = pd.read_csv(folder+ "gestationalweek.csv", delimiter=',', header=0)
gestationalweek = gestationalweek.sort_values(['person_id','condition_start_date'], ascending=True)
#stat concept_name
gestationalweek[['person_id']].groupby(gestationalweek['concept_name']).agg(['count'])#'mean', 
#remove concept_name: Abnormal/Normal pregnancy, First trimester pregnancy, Second trimester pregnancy, Third trimester pregnancy
gestationalweek = gestationalweek[~gestationalweek['concept_name'].isin(['Abnormal pregnancy', 'First trimester pregnancy','Normal pregnancy','Second trimester pregnancy','Third trimester pregnancy'])]

replace_value = {'Gestation period, 10 weeks': 10, 'Gestation period, 11 weeks': 11, 
                 'Gestation period, 12 weeks': 12, 'Gestation period, 13 weeks': 13,
                 'Gestation period, 14 weeks': 14, 'Gestation period, 15 weeks': 15,
                 'Gestation period, 16 weeks': 16, 'Gestation period, 17 weeks': 17,
                 'Gestation period, 18 weeks': 18, 'Gestation period, 19 weeks': 19,
                 'Gestation period, 20 weeks': 20, 'Gestation period, 21 weeks': 21,
                 'Gestation period, 2 weeks': 2, 'Gestation period, 22 weeks': 22,
                 'Gestation period, 23 weeks': 23, 'Gestation period, 24 weeks': 24,
                 'Gestation period, 25 weeks': 25, 'Gestation period, 26 weeks': 26,
                 'Gestation period, 27 weeks': 27, 'Gestation period, 28 weeks': 28,
                 'Gestation period, 29 weeks': 29, 'Gestation period, 30 weeks': 30,
                 'Gestation period, 3 weeks': 3, 'Gestation period, 31 weeks': 31,
                 'Gestation period, 32 weeks': 32, 'Gestation period, 33 weeks': 33,
                 'Gestation period, 34 weeks': 34, 'Gestation period, 35 weeks': 35,
                 'Gestation period, 36 weeks': 36, 'Gestation period, 37 weeks': 37,
                 'Gestation period, 38 weeks': 38, 'Gestation period, 39 weeks': 39,
                 'Gestation period, 40 weeks': 40, 'Gestation period, 4 weeks': 4,
                 'Gestation period, 41 weeks': 41, 'Gestation period, 42 weeks': 42,
                 'Gestation period, 5 weeks': 5, 'Gestation period, 6 weeks': 6,
                 'Gestation period, 7 weeks': 7, 'Gestation period, 8 weeks': 8,
                 'Gestation period, 9 weeks': 9}
gestationalweek['concept_name'] = gestationalweek['concept_name'].map(replace_value)
gestationalweek[['person_id']].groupby(gestationalweek['concept_name']).agg(['count'])#'mean',


#print(gestationalweek.condition_start_date.min())
#print(gestationalweek.condition_start_date.max())
gestationalweek = gestationalweek.drop_duplicates(['person_id','condition_start_date'])

gestationalweek = gestationalweek.sort_values(['person_id','condition_start_date'], ascending=True)
gestationalweek['condition_start_date_lag'] = gestationalweek.groupby(['person_id'])['condition_start_date'].transform(lambda x:x.shift())
gestationalweek['diffday'] = pd.DataFrame(pd.to_datetime(gestationalweek['condition_start_date']) - pd.to_datetime(gestationalweek['condition_start_date_lag']))
gestationalweek['diffday'] = gestationalweek['diffday'].dt.days
plt.hist(gestationalweek['diffday'], bins=50, color='steelblue', normed=True)
gestationalweek.shape#234978
gestationalweek.describe()
#include more data (based on 20w (pretem birth week) 20*7)
gestationalweek_1 = gestationalweek[gestationalweek['diffday']>140]
gestationalweek_2 = gestationalweek[(gestationalweek['diffday'].isnull())] #| (df['b'].isnull()) 
#nan is the first time pregnancy, >140 is the potential another delivery
gestationalweek = pd.concat([gestationalweek_1,gestationalweek_2],axis = 0)
gestationalweek = gestationalweek.sort_values(['person_id','condition_start_date'], ascending=True)
gestationalweek.shape#(37498, 30)
gestationalweek = gestationalweek.rename({'condition_start_date': 'ges_date','concept_name':'gweek'}, axis=1)
gestationalweek = pd.DataFrame(gestationalweek, columns=['person_id','ges_date','gweek'])
#person_id, ges_date,gweek
#gweek is the gestational week at ges_date


##############################################
#merge delivery2& gestationalweek
deliverygw = pd.merge(delivery2,gestationalweek,on = ['person_id'], how='inner')
deliverygw['day'] = pd.DataFrame(pd.to_datetime(deliverygw['condition_start_date']) - pd.to_datetime(deliverygw['ges_date']))#time difference between delivery and gestationalweek
deliverygw['day'] = deliverygw['day'].dt.days
# the day of the code of gw should be within 315(45*7) days of the delivery date
deliverygw = deliverygw[(deliverygw['day']>=0) & (deliverygw['day']<315)]#21023
deliverygw = deliverygw.sort_values(['person_id','condition_start_date','gweek'], ascending=True)
deliverygw = deliverygw.drop_duplicates(subset=['person_id','condition_start_date'], keep='last')

#fmyz = gestational week
deliverygw['fmyz'] = deliverygw['gweek'] + deliverygw['day']/7#17710
#limit to 20-46
deliverygw = deliverygw[(deliverygw['fmyz']>=20) & (deliverygw['fmyz']<46)]# 17633

##############################################
#person
person = pd.read_csv(folder+ "person.csv", delimiter=',', header=0)
#from datetime import datetime
#person['birthday'] = person.apply(lambda row: datetime(row['year_of_birth'], row['month_of_birth'], row['day_of_birth']), axis=1)
person = person.rename({'year_of_birth': 'Year', 'month_of_birth':'Month', 'day_of_birth':'Day'}, axis=1)
person['birthday'] = pd.to_datetime(person[['Year','Month','Day']])

#merge with deliverygw
deliverygw2 = pd.merge(deliverygw,person,on = ['person_id'], how='left')
#calcualte age（delivery date minus birthday）
deliverygw2['age'] = (pd.to_datetime(deliverygw2['condition_start_date']) - pd.to_datetime(deliverygw2['birthday']))/365.2425
deliverygw2['age'] = deliverygw2['age'].dt.days#20838
#limit age 18-45
deliverygw2 = deliverygw2[deliverygw2['age']>=18]
deliverygw2 = deliverygw2[deliverygw2['age']<=45]#17509

##############################################
#ppd -from 1w - 1y after childbirth
ppd = pd.read_csv(folder+ "ppd0820.csv", delimiter=',', header=0)
ppd[['mrn']].groupby(ppd['concept_name']).agg(['count'])#'mean',
ppd[['mrn']].groupby(ppd['concept_code']).agg(['count'])#'mean',
ppd = ppd.sort_values(['mrn','condition_start_date'], ascending=True)
ppd = ppd.drop_duplicates(subset=['mrn','condition_start_date'], keep='first')

ppd = pd.DataFrame(ppd, columns=['mrn','condition_start_date','concept_name'])
ppd['ppd'] = 1
ppd = ppd.rename({'condition_start_date': 'ppd_date'}, axis=1) #'a': 'X', 'b': 'Y'
ppd = pd.merge(deliverygw2,ppd,on = ['mrn'], how='inner')
ppd['diffppd'] = pd.DataFrame(pd.to_datetime(ppd['ppd_date']) - pd.to_datetime(ppd['condition_start_date']))
ppd['diffppd'] = ppd['diffppd'].dt.days
ppd = ppd[(ppd['diffppd'] >=0 ) & (ppd['diffppd'] < 367)]
#plt.hist(ppd['diffppd'],bins=366,color='steelblue') # bins=50, #normed=1,probablity
#plt.xlabel('days after childbirth')  
#plt.ylabel('Frequency') 
#plt.show()

ppd = ppd.drop_duplicates(subset=['mrn','condition_start_date'], keep='first')
ppd[['person_id']].groupby(ppd['concept_name_y']).agg(['count'])#'mean',
ppd = pd.DataFrame(ppd, columns=['person_id','condition_start_date','ppd_date','ppd','diffppd'])
deliverygw2 = pd.merge(deliverygw2,ppd,on = ['person_id','condition_start_date'], how='left')
deliverygw2['ppd']=deliverygw2['ppd'].fillna(0)
deliverygw2['ppd'].value_counts()

#                                                       count
#concept_name_y                                              
#Adjustment disorder with depressed mood                    7
#Anxiety                                                   58
#Anxiety disorder                                          53
#Anxiety state                                             42
#Depressed mood                                            10
#Depressed mood with postpartum onset                       2
#Depressive disorder in mother complicating preg...         1
#Generalized anxiety disorder                              45
#Major depression, single episode                          41
#Major depressive disorder                                  1
#Mild postnatal depression                                  4
#Postpartum depression                                     90
#Recurrent major depression                                 6
#Severe postnatal depression                                1
#Severe recurrent major depression without psych...         3
#Single episode of major depression in full remi...         1

##############################################
#antidepressants to define ppd
#include all codes under N06A, and then exclude the following, which are likely to have a dual indication for prescribing, in addition to mental health indication:
#N06AA09 Amitriptyline,Tryptizol
#N06AA04 Anafranil, Clomipramine
#N06AX21 Cymbalta,Duloxetine,Yentreve
#N05AF01 Fluanxol,Flupentixol
#N06AA10 Nortriptyline
#The reference for this approach is: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4828938/
drug14 = pd.read_csv(folder+ "drug14.csv", delimiter=',', header=0)
drug15 = pd.read_csv(folder+ "drug15.csv", delimiter=',', header=0)
drug16 = pd.read_csv(folder+ "drug16.csv", delimiter=',', header=0)
drug17 = pd.read_csv(folder+ "drug17.csv", delimiter=',', header=0)
drug18 = pd.read_csv(folder+ "drug18.csv", delimiter=',', header=0)
drug19 = pd.read_csv(folder+ "drug19.csv", delimiter=',', header=0)
medication = pd.concat([drug14,drug15,drug16,drug17,drug18,drug19],axis = 0)

#medication.head(1)
# String to be searched in start of string  
search ="N06A"
# series returned with False at place of NaN 
series = medication['concept_code.1'].str.startswith(search, na = False) 
# displaying filtered dataframe 
antidepressant = medication[series] #18129
#or
#antidepressant = medication[medication['concept_code.1'].str.contains('N06A')]
antidepressant['concept_class_id.1'].value_counts()
#the number of 3rd 4th 5th are the same 6043
#then exclude codes before
antidepressant3 = antidepressant.loc[antidepressant['concept_class_id.1'].isin(['ATC 5th'])]
antidepressant3 = antidepressant3[~antidepressant3['concept_code.1'].isin(['N06AA09', 'N06AA04','N06AX21','N05AF01','N06AA10'])]
antidepressant3['concept_name.1'].value_counts()  
#sertraline        1728
#escitalopram      1174
#fluoxetine         648
#bupropion          615
#trazodone          411
#citalopram         322
#venlafaxine        299
#mirtazapine        232
#paroxetine          77
#fluvoxamine         50
#desvenlafaxine      16
#vortioxetine        14
#doxepin             12
#milnacipran          4
#vilazodone           4
#desipramine          3
#protriptyline        2
#amoxapine            1
#oxitriptan           1
#imipramine           1

antidepressant3 = pd.merge(deliverygw2,antidepressant3,on = ['mrn'], how='inner')
#calcualte age
antidepressant3['diffanti'] = pd.to_datetime(antidepressant3['drug_exposure_start_date']) - pd.to_datetime(antidepressant3['condition_start_date'])
antidepressant3['diffanti'] = antidepressant3['diffanti'].dt.days
antidepressant3 = antidepressant3[(antidepressant3['diffanti'] >=0 ) & (antidepressant3['diffanti'] < 367)]
antidepressant3 = antidepressant3.drop_duplicates(subset=['mrn','condition_start_date'], keep='first')
antidepressant3['ppd'].value_counts()
antidepressant3 = pd.DataFrame(antidepressant3, columns=['mrn','condition_start_date','drug_exposure_start_date','diffanti','ppd'])
antidepressant3 = antidepressant3[(antidepressant3['ppd'] == 0 )]
antidepressant3 = antidepressant3.drop('ppd', 1)#0 for rows and 1 for columns
#merge deliverygw2
#diffanti, drug_exposure_start_date
#diffppd, ppd_date
deliverygw3 = pd.merge(deliverygw2,antidepressant3,on = ['mrn','condition_start_date'], how='left')
#deliverygw3.loc[deliverygw3['diffppd'].isnull(),'diffppd']=deliverygw3[deliverygw3['diffppd'].isnull()]['diffanti']
deliverygw3.loc[deliverygw3['diffppd'].isnull(),'diffppd']=deliverygw3['diffanti']
#deliverygw3.loc[deliverygw3['ppd_date'].isnull(),'ppd_date']=deliverygw3[deliverygw3['ppd_date'].isnull()]['drug_exposure_start_date']
deliverygw3.loc[deliverygw3['ppd_date'].isnull(),'ppd_date']=deliverygw3['drug_exposure_start_date']

deliverygw3.ppd[deliverygw3.diffppd > 0] = 1
deliverygw3 = deliverygw3.drop('drug_exposure_start_date', 1)
deliverygw3['ppd'].value_counts()


#plt.hist(deliverygw3['diffppd'],bins=366,color='steelblue') # bins=50, #normed=1,probablity
#plt.xlabel('days after childbirth(WCM)')  
#plt.ylabel('Frequency') 
#plt.show()

##############################################
#condition
deliverygw4 = deliverygw3
deliverygw4['gestationday'] = deliverygw4['fmyz'] * 7
condition = pd.read_csv(folder+ "condition.csv", delimiter=',', header=0)
#condition['person_id'].groupby(condition['condition_type_concept_id']).agg(['count'])
condition = condition.sort_values(['person_id','condition_start_date','concept_name'], ascending=True)
condition = condition.drop_duplicates(subset=['person_id','condition_start_date','concept_name'], keep='first')
condition = condition.rename({'condition_start_date': 'condition_date'}, axis=1) 
condition = pd.merge(deliverygw4,condition,on = ['person_id'], how='inner')
condition['diffcondition'] = pd.to_datetime(condition['condition_start_date']) - pd.to_datetime(condition['condition_date'])
condition['diffcondition'] = condition['diffcondition'].dt.days

condition = condition[(deliverygw4['gestationday'] - condition['diffcondition']) >= 0] 
condition = condition[(deliverygw4['gestationday'] - condition['diffcondition']) <= 126] #18*7

condition = condition.drop_duplicates(subset=['person_id','condition_start_date','concept_name_y'], keep='first')
condition = pd.DataFrame(condition, columns=['person_id','condition_start_date','condition_date','concept_name_y','diffcondition'])

condition_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/condition_listcombine.csv', delimiter=',', index_col=0, header=0)
#condition_list = condition_list.rename({'concept_name': 'concept_name_y'}, axis=1) #'a': 'X', 'b': 'Y'

#!condition_list
condition = pd.merge(condition,condition_list,on = ['concept_name_y'], how='left')
condition.isnull().sum(axis=0)
#null_data = condition[condition.isnull().any(axis=1)]#output null data
condition2 = pd.DataFrame(condition, columns=['person_id','condition_start_date','newvariable'])
condition2 = condition2.drop_duplicates(subset=['person_id','condition_start_date','newvariable'], keep='first')
#pivot_table() to transform the data 
condition2['values'] = 1
condition2 = (condition2.pivot_table(index=['person_id','condition_start_date'], columns='newvariable',values='values').reset_index())

##############################################
#medication
#medication = pd.concat([drug14,drug15,drug16,drug17,drug18,drug19],axis = 0)
medication2 = medication.loc[medication['concept_class_id.1'].isin(['ATC 3rd'])]
medication2 = medication2.sort_values(['person_id','drug_exposure_start_date','concept_name.1'], ascending=True)
medication2 = medication2.drop_duplicates(subset=['person_id','drug_exposure_start_date','concept_name.1'], keep='first')
medication2 = pd.DataFrame(medication2, columns=['person_id','drug_exposure_start_date','concept_name.1'])

#del deliverygw4['drug_exposure_start_date']

medication2 = pd.merge(deliverygw4,medication2,on = ['person_id'], how='inner')
medication2['diffmedication'] = pd.to_datetime(medication2['condition_start_date']) - pd.to_datetime(medication2['drug_exposure_start_date'])
medication2['diffmedication'] = medication2['diffmedication'].dt.days

medication2 = medication2[(medication2['gestationday'] - medication2['diffmedication']) >= 0] 
medication2 = medication2[(medication2['gestationday'] - medication2['diffmedication']) <= 126] #18*7

medication2 = medication2.drop_duplicates(subset=['person_id','drug_exposure_start_date','concept_name.1'], keep='first')
medication2 = pd.DataFrame(medication2, columns=['person_id','drug_exposure_start_date','condition_start_date','concept_name.1','diffmedication'])

#!medication_list
medication_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/medication_listnew.csv',delimiter=',', header=0)
medication2 = pd.merge(medication2,medication_list,on = ['concept_name.1'], how='left')
#print(medication2.head(2))
medication2.isnull().sum(axis=0)

medication2 = pd.DataFrame(medication2, columns=['person_id','condition_start_date','newvariable'])
medication2 = medication2.drop_duplicates(subset=['person_id','condition_start_date','newvariable'], keep='first')
medication2.groupby('newvariable').count()
#pivot_table() to transform the data (same with cast/dcast in R)
medication2['values'] = 1
medication2 = (medication2.pivot_table(index=['person_id','condition_start_date'], columns='newvariable',values='values').reset_index())

##############################################
#pre-pregnancy BMI
bmi = pd.read_csv(folder+ "bmi.csv", delimiter=',', header=0)
bmi = bmi[(bmi['value_source_value'] >10 ) & (bmi['value_source_value'] < 60)]
bmi = pd.DataFrame(bmi, columns=['person_id','measurement_date','value_as_number'])
bmi = pd.merge(deliverygw4,bmi,on = ['person_id'], how='inner')
bmi['diffbmi'] = pd.to_datetime(bmi['condition_start_date']) - pd.to_datetime(bmi['measurement_date'])
bmi['diffbmi'] = bmi['diffbmi'].dt.days-bmi['gestationday']
bmi = bmi.sort_values(['person_id','condition_start_date','measurement_date'], ascending=True)
bmi['diffbmi'] = np.abs(bmi['diffbmi'])
bmi = bmi.sort_values(['person_id','condition_start_date','diffbmi']).groupby(['person_id','condition_start_date'], as_index=False).first()
bmi = pd.DataFrame(bmi, columns=['person_id','condition_start_date','value_as_number'])
bmi = bmi.rename({'value_as_number': 'bmi'}, axis=1) 
##############################################
#bp
bp = pd.read_csv(folder+ "bp.csv", delimiter=',', header=0)
plt.hist(bp['value_source_value'], bins=50, color='steelblue', normed=True)
bp = bp[(bp['value_source_value'] >30 )]
bp = pd.DataFrame(bp, columns=['person_id','measurement_date','value_as_number','concept_name'])
bp = pd.merge(deliverygw4,bp,on = ['person_id'], how='inner')
bp['diffbp'] = pd.to_datetime(bp['condition_start_date']) - pd.to_datetime(bp['measurement_date'])
bp['diffbp'] = bp['diffbp'].dt.days
bp = bp[(bp['diffbp'] >= 0 ) & (bp['diffbp'] <= bp['gestationday'])]

def trimester(bp):
    if ((bp['gestationday'] - bp['diffbp']) < 84):
        return 1
    if ((bp['gestationday'] - bp['diffbp']) >=84) and ((bp['gestationday'] - bp['diffbp']) < 126):
        return 2
    else:
        return 3
bp['trim'] = bp.apply(trimester, axis=1)

sbp = bp[(bp['concept_name_y'] == 'BP systolic')]
dbp = bp[(bp['concept_name_y'] == 'BP diastolic')]

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
weight = pd.read_csv(folder+ "weight.csv", delimiter=',', header=0)
#Remove strings from a float number in a column
weight['weight2'] = pd.to_numeric(weight.value_source_value.str.replace('[^\d.]', ''), errors='coerce')
#plt.hist(weight['weight2'], bins=50, color='steelblue', normed=True)
weight = weight[(weight['weight2'] >20 ) & (weight['weight2'] < 300)]
#plt.hist(weight['weight2'], bins=50, color='steelblue', normed=True)
weight = pd.DataFrame(weight, columns=['person_id','measurement_date','weight2'])
weight = pd.merge(deliverygw4,weight,on = ['person_id'], how='inner')
weight['diffweight'] = pd.to_datetime(weight['condition_start_date']) - pd.to_datetime(weight['measurement_date'])
weight['diffweight'] = weight['diffweight'].dt.days
weight = weight[(weight['diffweight'] >= 0 ) & (weight['diffweight'] <= weight['gestationday'])]

def trimester(weight):
    if ((weight['gestationday'] - weight['diffweight']) < 84):
        return 1
    if ((weight['gestationday'] - weight['diffweight']) >=84) and ((weight['gestationday'] - weight['diffweight']) < 126):
        return 2
    else:
        return 3
weight['trim'] = weight.apply(trimester, axis=1)

weight['weightmean'] = weight.groupby(['person_id','condition_start_date','trim'])['weight2'].transform('mean')
weight = pd.DataFrame(weight, columns=['person_id','condition_start_date','weightmean','trim'])
weight = weight.drop_duplicates(subset=['person_id','condition_start_date','weightmean','trim'], keep='first')

weight = (weight.pivot_table(index=['person_id','condition_start_date'], columns='trim',values='weightmean').reset_index())
weight = weight.rename({1: 'weight1st',2: 'weight2nd',3: 'weight3rd'}, axis=1) #'a': 'X', 'b': 'Y'

##############################################
#amniocentesis in second trimester
amniocentesis = pd.read_csv(folder+ "amniocentesis.csv", delimiter=',', header=0)
amniocentesis.groupby(['concept_name','concept_id']).count()
amniocentesis = amniocentesis[~amniocentesis['concept_id'].isin(['2110331','2110332'])]
amniocentesis = pd.DataFrame(amniocentesis, columns=['person_id','procedure_date'])
amniocentesis = pd.merge(deliverygw4,amniocentesis,on = ['person_id'], how='inner')
amniocentesis['diffamniocentesis'] = pd.to_datetime(amniocentesis['condition_start_date']) - pd.to_datetime(amniocentesis['procedure_date'])
amniocentesis['diffamniocentesis'] = amniocentesis['diffamniocentesis'].dt.days

amniocentesis = amniocentesis[(amniocentesis['diffamniocentesis'] >= 0 ) & (amniocentesis['diffamniocentesis'] <= amniocentesis['gestationday'])]
amniocentesis = amniocentesis.drop_duplicates(subset=['person_id','procedure_date'], keep='first')
amniocentesis = pd.DataFrame(amniocentesis, columns=['person_id','condition_start_date'])
amniocentesis['amniocentesis'] = 1
amniocentesis = amniocentesis.drop_duplicates()

##############################################
#marital
married = pd.read_csv(folder+ "Married.csv", delimiter=',', header=0)
single = pd.read_csv(folder+ "single.csv", delimiter=',', header=0)
widowed = pd.read_csv(folder+ "WIDOWED.csv", delimiter=',', header=0)

divorced1 = pd.read_csv(folder+ "DIVORCED2012.csv", delimiter=',', header=0)
divorced2 = pd.read_csv(folder+ "DIVORCED2016.csv", delimiter=',', header=0)
divorced3 = pd.read_csv(folder+ "DIVORCED2017.csv", delimiter=',', header=0)
divorced4 = pd.read_csv(folder+ "DIVORCED2018.csv", delimiter=',', header=0)
divorced5 = pd.read_csv(folder+ "DIVORCED20131.csv", delimiter=',', header=0)
divorced6 = pd.read_csv(folder+ "DIVORCED20132.csv", delimiter=',', header=0)
divorced7 = pd.read_csv(folder+ "DIVORCED20133.csv", delimiter=',', header=0)
divorced8 = pd.read_csv(folder+ "DIVORCED20134.csv", delimiter=',', header=0)
divorced9 = pd.read_csv(folder+ "DIVORCED20135.csv", delimiter=',', header=0)
divorced10 = pd.read_csv(folder+ "DIVORCED20141.csv", delimiter=',', header=0)
divorced11 = pd.read_csv(folder+ "DIVORCED20142.csv", delimiter=',', header=0)
divorced12 = pd.read_csv(folder+ "DIVORCED20144.csv", delimiter=',', header=0)
divorced13 = pd.read_csv(folder+ "DIVORCED20145.csv", delimiter=',', header=0)
divorced14 = pd.read_csv(folder+ "DIVORCED20147.csv", delimiter=',', header=0)
divorced15 = pd.read_csv(folder+ "DIVORCED20148.csv", delimiter=',', header=0)
divorced16 = pd.read_csv(folder+ "DIVORCED20149.csv", delimiter=',', header=0)
divorced17 = pd.read_csv(folder+ "DIVORCED201410.csv", delimiter=',', header=0)
divorced18 = pd.read_csv(folder+ "DIVORCED201412.csv", delimiter=',', header=0)
divorced19 = pd.read_csv(folder+ "DIVORCED201501.csv", delimiter=',', header=0)
divorced20 = pd.read_csv(folder+ "DIVORCED201504.csv", delimiter=',', header=0)
divorced21 = pd.read_csv(folder+ "DIVORCED201506.csv", delimiter=',', header=0)
divorced22 = pd.read_csv(folder+ "DIVORCED201507.csv", delimiter=',', header=0)
divorced23 = pd.read_csv(folder+ "DIVORCED201509.csv", delimiter=',', header=0)
divorced24 = pd.read_csv(folder+ "DIVORCED201510.csv", delimiter=',', header=0)
divorced25 = pd.read_csv(folder+ "DIVORCED201511.csv", delimiter=',', header=0)
divorced26 = pd.read_csv(folder+ "DIVORCED201512.csv", delimiter=',', header=0)
divorced = pd.concat([divorced1,divorced2,divorced3,divorced4,divorced5,divorced6,divorced7,divorced8,divorced9,divorced10,divorced11,divorced12,divorced13,divorced14,divorced15,divorced16,divorced17,divorced18,divorced19,divorced20,divorced21,divorced22,divorced23,divorced24,divorced25,divorced26],axis = 0)

married['marital'] = 1
single['marital'] = 2
widowed['marital'] = 2
divorced['marital'] = 2
marital = pd.concat([married,single,widowed,divorced],axis = 0)
marital = pd.merge(deliverygw4,marital,on = ['person_id'], how='inner')
marital['diffmarital'] = pd.to_datetime(marital['condition_start_date']) - pd.to_datetime(marital['note_date'])
marital['diffmarital'] = marital['diffmarital'].dt.days
marital = marital.drop_duplicates(subset=['person_id','condition_start_date','marital'], keep='first')

marital['diffmarital'] = np.abs(marital['diffmarital'])
marital = marital.sort_values(['person_id','condition_start_date','diffmarital']).groupby(['person_id','condition_start_date'], as_index=False).first()
marital = pd.DataFrame(marital, columns=['person_id','condition_start_date','marital'])


##############################################
#visitoccurrence edvisit
visit = pd.read_csv(folder+ "visit.csv", delimiter=',', header=0)
visit = visit[visit['visit_concept_id'].isin(['262','9203'])]
visit = visit.drop_duplicates(subset=['person_id','visit_start_date'], keep='first')

visit = pd.merge(deliverygw4,visit,on = ['person_id'], how='inner')
visit['diffvisit'] = pd.to_datetime(visit['condition_start_date']) - pd.to_datetime(visit['visit_start_date'])
visit['diffvisit'] = visit['diffvisit'].dt.days
visit = visit[(visit['gestationday'] - visit['diffvisit']) < 126] 

visit['value'] = 1
visit = visit.groupby(['person_id','condition_start_date'])['value'].count().reset_index(name="edvisitcount")


##############################################
#merge data to build the model file
def race(a):
    if (a['race_source_value'] == 'ASHKENAZI JEWISH') or (a['race_source_value'] == 'NAT.HAWAIIAN/OTH.PACIFIC ISLAND') or (a['race_source_value'] =='WHITE'):
        return 1
    if (a['race_source_value'] == 'ASIAN') or (a['race_source_value'] == 'ASIAN INDIAN'):
        return 2
    if (a['race_source_value'] == 'AMERICAN INDIAN OR ALASKA NATION' or a['race_source_value'] == 'BLACK OR AFRICAN AMERICAN'):
        return 3
    if (a['race_source_value'] == 'OTHER COMBINATIONS NOT DESCRIBED'):
        return 4
    else:
        return 5
deliverygw4['race2'] = deliverygw4.apply(race, axis=1)

demo = pd.DataFrame(deliverygw4, columns=['person_id','condition_start_date','fmyz','race2','age','ppd'])


wcmmodel = pd.merge(demo,marital,on = ['person_id','condition_start_date'], how='left')
wcmmodel['marital']=wcmmodel['marital'].fillna(1)
wcmmodel = pd.merge(wcmmodel,visit,on = ['person_id','condition_start_date'], how='left')
wcmmodel['edvisitcount']=wcmmodel['edvisitcount'].fillna(0)
wcmmodel = pd.merge(wcmmodel,amniocentesis,on = ['person_id','condition_start_date'], how='left')
wcmmodel['amniocentesis']=wcmmodel['amniocentesis'].fillna(0)

#wcmmodel = pd.merge(wcmmodel,insurance,on = ['person_id','condition_start_date'], how='left')

wcmmodel = pd.merge(wcmmodel,weight,on = ['person_id','condition_start_date'], how='left')
wcmmodel = pd.merge(wcmmodel,bp,on = ['person_id','condition_start_date'], how='left')
wcmmodel = pd.merge(wcmmodel,bmi,on = ['person_id','condition_start_date'], how='left')
wcmmodel = wcmmodel.fillna(wcmmodel.mean())

#Normalize data
#normalize = pd.DataFrame(wcmmodel, columns=['fmyz','age','weight1st','weight2nd','weight3rd','sbp1st','sbp2nd','sbp3rd','dbp1st','dbp2nd','dbp3rd'])
num_cols = wcmmodel[['fmyz','age','edvisitcount','weight1st','weight2nd','weight3rd','sbp1st','sbp2nd','sbp3rd','dbp1st','dbp2nd','dbp3rd','bmi']]
scaler = StandardScaler()
num_cols = scaler.fit_transform(num_cols)
normalize = pd.DataFrame(num_cols)

normalize.rename(columns={1:'age',
                          2:'edvisitcount',
                          3:'weight1st',
                          4:'weight2nd',
                          0:'fmyz',
                          5:'weight3rd',
                          6:'sbp1st',
                          7:'sbp2nd',
                          8:'sbp3rd',
                          9:'dbp1st',
                          10:'dbp2nd',
                          11:'dbp3rd',
                          12:'bmi'
                          }, 
                 inplace=True)

wcmmodel['fmyz'] = normalize['fmyz']
wcmmodel['edvisitcount'] = normalize['edvisitcount']
wcmmodel['weight1st'] = normalize['weight1st']
wcmmodel['weight2nd'] = normalize['weight2nd']
wcmmodel['age'] = normalize['age']
wcmmodel['weight3rd'] = normalize['weight3rd']
wcmmodel['sbp1st'] = normalize['sbp1st']
wcmmodel['sbp2nd'] = normalize['sbp2nd']
wcmmodel['sbp3rd'] = normalize['sbp3rd']
wcmmodel['dbp1st'] = normalize['dbp1st']
wcmmodel['dbp2nd'] = normalize['dbp2nd']
wcmmodel['dbp3rd'] = normalize['dbp3rd']
wcmmodel['bmi'] = normalize['bmi']

# Get one hot encoding of columns race2 marital
one_hotrace = pd.get_dummies(wcmmodel['race2'])
#1:white
#2:asian
#3:black
#4:combine
#5:other
one_hotrace = one_hotrace.rename({1: 'white',2:'asian',3:'black',4:'combine',5:'otherrace'}, axis=1) #'a': 'X', 'b': 'Y'
wcmmodel = wcmmodel.drop('race2',axis = 1)
wcmmodel = wcmmodel.join(one_hotrace)

one_hotmarital = pd.get_dummies(wcmmodel['marital'])

one_hotmarital = one_hotmarital.rename({1: 'married',2:'single'}, axis=1) #'a': 'X', 'b': 'Y'
wcmmodel = wcmmodel.drop('marital',axis = 1)
wcmmodel = wcmmodel.join(one_hotmarital)
#remove variables in third trimester
wcmmodel = wcmmodel.drop(['weight3rd','sbp3rd','dbp3rd'], axis=1)


#merge with diagnose/medication
wcmmodel = pd.merge(wcmmodel,condition2,on = ['person_id','condition_start_date'], how='left')
wcmmodel = pd.merge(wcmmodel,medication2,on = ['person_id','condition_start_date'], how='left')
wcmmodel = wcmmodel.fillna(0)
#wcmmodel.to_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelnohistory.csv', sep=',',index=0)


##############################################
#add history

def func(deliverygw4, disease="anxiety", folder="/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/wcm/history/"):
    del4 = deliverygw4
    file_dir = folder+disease+".csv"
    data = pd.read_csv(file_dir, delimiter=',', header=0)
    data = data.sort_values(['person_id','condition_start_date'], ascending=True)
    data = data.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
    data[disease] = 1
    data = data.rename({'condition_start_date': disease+"_date"}, axis=1) #'a': 'X', 'b': 'Y'
    data = pd.merge(del4,data,on = ['person_id'], how='inner')
    data["diff"+disease] = pd.DataFrame(pd.to_datetime(data[disease+"_date"]) - pd.to_datetime(data['condition_start_date']))
    data["diff"+disease] = data["diff"+disease].dt.days
    data = data[(data["diff"+disease] > 0 )]
    data = data.drop_duplicates(subset=['person_id','condition_start_date'], keep='first')
    data = pd.DataFrame(data, columns=['person_id','condition_start_date',disease])
    return data


def main():
    global new_deliverygw4
       
    disease_list = ["anxiety", "mooddisorder", "organicdisorder", "otherdisorder", "personalitydisorder", "schizophrenic", "substance"]
    #disease_list = ["anxiety", "mooddisorder"]
    
    new_deliverygw4 = deliverygw4
    for d in disease_list:
        disease_table = func(deliverygw4, disease=d)
        new_deliverygw4 = pd.merge(new_deliverygw4,disease_table,on = ['person_id','condition_start_date'], how='left')
#        print(disease_table)
    #print(new_deliverygw4)

if __name__ == "__main__":
    main()

#the frequency of personalitydisorder is below 10

history = pd.DataFrame(new_deliverygw4, columns=['person_id','condition_start_date',"anxiety", "mooddisorder", "organicdisorder", "otherdisorder","schizophrenic", "substance"])
history = history.fillna(0)
wcmmodelhistory = pd.merge(wcmmodel,history,on = ['person_id','condition_start_date'], how='left')
print(wcmmodelhistory.isnull().sum())
wcmmodelhistory['ppd'].value_counts()

wcmmodelhistory.to_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18_2.csv', sep=',',index=0)





