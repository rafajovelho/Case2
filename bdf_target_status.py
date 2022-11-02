#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import pandas as pd
import pymysql
import requests
import io
import xlsxwriter
import calendar
import numpy as np

import datetime


# In[2]:


def connect(host,user, password):
    conn = pymysql.connect(
        host=host,
        port=int(3306),
        user=user,
        passwd=password,
                           )
    return conn

def beiersdorf(host,user, password):
    
    _,num_days_prev = calendar.monthrange(datetime.datetime.today().year, datetime.datetime.today().month)
    _, num_days = calendar.monthrange(datetime.datetime.today().year, datetime.datetime.today().month+1)
    
    current_month_start = datetime.date(datetime.datetime.today().year,datetime.datetime.today().month+1,1).strftime('%Y-%m-%d')
    current_month_end   = datetime.date(datetime.datetime.today().year,datetime.datetime.today().month+1,num_days).strftime('%Y-%m-%d')
    prev_month_start    = datetime.date(datetime.datetime.today().year,datetime.datetime.today().month,1).strftime('%Y-%m-%d')
    prev_month_end      = datetime.date(datetime.datetime.today().year,datetime.datetime.today().month,num_days_prev).strftime('%Y-%m-%d')

    conn =  connect(host,user, password)
    KPIS = pd.read_sql_query("""
SELECT kpi.type                                               as "KPI"
 ,SUM(CASE WHEN (COALESCE(ext.start_date, ASS_PRODUCT.start_date) <= '{}' AND COALESCE(ext.end_date, ASS_PRODUCT.end_date) >= '{}') THEN 1 ELSE 0 END)  as "Current month targets"
 ,SUM(CASE WHEN (COALESCE(ext.start_date, ASS_PRODUCT.start_date) <= '{}' AND COALESCE(ext.end_date, ASS_PRODUCT.end_date) >= '{}') THEN 1 ELSE 0 END) as "Previous month targets"
 ,MAX(ext.received_time)                                            as "Last Updated"
FROM static.kpi_level_2 kpi
LEFT JOIN static.kpi_external_targets ext on kpi.pk = ext.kpi_level_2_fk
LEFT JOIN pservice.assortment as ASS ON ASS.kpi_fk = kpi.pk
LEFT JOIN pservice.assortment_to_product ASS_PRODUCT ON ASS_PRODUCT.assortment_fk = ASS.pk
WHERE YEAR(COALESCE(ext.end_date, ASS_PRODUCT.end_date)) >= 2022 AND kpi.type NOT IN ('Number Of Secondary Placement by Type','Secondary Shelf')
GROUP BY 1""".format(current_month_end,current_month_start,prev_month_end,prev_month_start),conn)
    secondary_branded = pd.read_sql_query("""
SELECT CONCAT(kpi.type," - Branded targets")                                               as "KPI"
 ,SUM(CASE WHEN (COALESCE(ext.start_date, ASS_PRODUCT.start_date) <= '{}' AND COALESCE(ext.end_date, ASS_PRODUCT.end_date) >= '{}') THEN ext.data_json ->> "$.sp_branded_target" ELSE 0 END)  as "Current month targets"
 ,SUM(CASE WHEN (COALESCE(ext.start_date, ASS_PRODUCT.start_date) <= '{}' AND COALESCE(ext.end_date, ASS_PRODUCT.end_date) >= '{}') THEN ext.data_json ->> "$.sp_branded_target" ELSE 0 END) as "Previous month targets"
 ,MAX(ext.received_time)                                            as "Last Updated"
FROM static.kpi_level_2 kpi
LEFT JOIN static.kpi_external_targets ext on kpi.pk = ext.kpi_level_2_fk
LEFT JOIN pservice.assortment as ASS ON ASS.kpi_fk = kpi.pk
LEFT JOIN pservice.assortment_to_product ASS_PRODUCT ON ASS_PRODUCT.assortment_fk = ASS.pk
WHERE YEAR(COALESCE(ext.end_date, ASS_PRODUCT.end_date)) >= 2022 AND kpi.type IN ('Number Of Secondary Placement by Type','Secondary Shelf')
GROUP BY 1""".format(current_month_end,current_month_start,prev_month_end,prev_month_start),conn)
    KPIS = KPIS.append(secondary_branded, ignore_index=True)
    secondary_sp = pd.read_sql_query("""
SELECT CONCAT(kpi.type," - SP targets")                                               as "KPI"
 ,SUM(CASE WHEN (COALESCE(ext.start_date, ASS_PRODUCT.start_date) <= '2022-06-30' AND COALESCE(ext.end_date, ASS_PRODUCT.end_date) >= '2022-06-01') THEN ext.data_json ->> "$.sp_target" ELSE 0 END)  as "Current month targets"
 ,SUM(CASE WHEN (COALESCE(ext.start_date, ASS_PRODUCT.start_date) <= '2022-05-31' AND COALESCE(ext.end_date, ASS_PRODUCT.end_date) >= '2022-05-01') THEN ext.data_json ->> "$.sp_target" ELSE 0 END) as "Previous month targets"
 ,MAX(ext.received_time)                                            as "Last Updated"
FROM static.kpi_level_2 kpi
LEFT JOIN static.kpi_external_targets ext on kpi.pk = ext.kpi_level_2_fk
LEFT JOIN pservice.assortment as ASS ON ASS.kpi_fk = kpi.pk
LEFT JOIN pservice.assortment_to_product ASS_PRODUCT ON ASS_PRODUCT.assortment_fk = ASS.pk
WHERE YEAR(COALESCE(ext.end_date, ASS_PRODUCT.end_date)) >= 2022 AND kpi.type IN ('Number Of Secondary Placement by Type','Secondary Shelf')
GROUP BY 1""".format(current_month_end,current_month_start,prev_month_end,prev_month_start),conn)

    
    
    KPIS = KPIS.append(secondary_sp, ignore_index=True)
    
    

    KPIS["Difference (#)"] = KPIS["Current month targets"] - KPIS["Previous month targets"]


    KPIS["Difference (%)"] = np.nan

    for i in range(0, len(KPIS["Previous month targets"])):
        if ((KPIS["Previous month targets"].loc[i] == 0) & (KPIS["Current month targets"].loc[i] == 0)):
            KPIS["Difference (%)"].loc[i] = 0
        elif ((KPIS["Previous month targets"].loc[i]) == 0 & (KPIS["Current month targets"].loc[i] != 0)):
            KPIS["Difference (%)"].loc[i] = 100
        else:
            KPIS["Difference (%)"].loc[i] = ((KPIS["Current month targets"][i]/KPIS["Previous month targets"][i])-1)*100


    KPIS["Difference (%)"] = KPIS["Difference (%)"].round(2)
    return KPIS


# In[3]:


project_id = input("Which project do you want to extract the targets from?")


# In[4]:


if project_id[:-2] == "beiersdorf": 
    kpis = beiersdorf("mysql.{}-prod.us-east-1.trax-cloud.com".format(project_id),"traxreadonly","2015Trax@DB")
    kpis["project"] = project_id
    writer = pd.ExcelWriter('{}_targets_status.xlsx'.format(project_id), engine='xlsxwriter')
    kpis.to_excel(writer, sheet_name="status",startrow=1, index=False)
    writer.save()
        
elif project_id == "bdftr": 
    kpis = beiersdorf("mysql.{}-prod.us-east-1.trax-cloud.com".format(project_id),"traxreadonly","2015Trax@DB")    
    kpis["project"] = project_id
    writer = pd.ExcelWriter('{}_targets_status.xlsx'.format(project_id), engine='xlsxwriter')
    kpis.to_excel(writer, sheet_name="status",startrow=1, index=False)
    writer.save()
        

elif project_id == "beiersdorfuae": 
    kpis = beiersdorf("mysql.{}-prod.us-east-1.trax-cloud.com".format(project_id),"traxreadonly","2015Trax@DB")    
    kpis["project"] = project_id
    writer = pd.ExcelWriter('{}_targets_status.xlsx'.format(project_id), engine='xlsxwriter')
    kpis.to_excel(writer, sheet_name="status",startrow=1, index=False)
    writer.save()

            
elif project_id == "all":
    kpis = pd.DataFrame()
    BDF = ['beiersdorfeg','beiersdorfid','beiersdorftw','beiersdorfar','beiersdorfgh','beiersdorfpt','beiersdorfke','beiersdorfkz','bdftr','beiersdorfng','beiersdorfru','beiersdorfsa','beiersdorfua','beiersdorfuae','beiersdorfza','beiersdorfbr','beiersdorfchl','beiersdorfco','beiersdorfcr','beiersdorfec','beiersdorfgt','beiersdorfmx','beiersdorfpa','beiersdorfpe','beiersdorfau','beiersdorfin','beiersdorfmy','beiersdorfph','beiersdorfth','beiersdorfvn','beiersdorfnz']
    for bdf in BDF:
        a = beiersdorf("mysql.{}-prod.us-east-1.trax-cloud.com".format(bdf),"traxreadonly","2015Trax@DB")
        a["project"] = bdf
        kpis = kpis.append(a)
    # kpis = kpis.transpose()
    writer = pd.ExcelWriter('bdf_targets_status.xlsx', engine='xlsxwriter')
    kpis.to_excel(writer, sheet_name="status",startrow=1, index=False)
    writer.save()


# In[ ]:





# In[ ]:





# In[ ]:





# In[62]:





# In[24]:





# In[25]:





# In[ ]:




