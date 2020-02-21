from QueryFunctions import * 
from queryrunner_client import Client 

start_date = '2019-05-25' 
end_date = '2019-08-10' 
city_ids = '5, 14, 12, 7, 8, 23, 198, 1, 25, 27, 134, 208, 10' 
#city_ids = '5, 14, 12, 7, 8, 23, 198, 1, 25, 27, 134, 208, 10, 1541, 26, 4, 15, 40, 190, 142, 207' 
bucket_cap1 = '4' 
bucket_cap2 = '5' 

qr = Client(user_email='will.zou@uber.com') 
query_results = qr.execute('presto-secure', hscls_model_data_dualcap(start_date, end_date, city_ids, bucket_cap1, bucket_cap2)) 

import pandas as pd 

query_results = pd.DataFrame(query_results.load_data()) 

print(len(query_results)) 

query_results.to_csv('/home/udocker/will.zou/sqldataproc/hs_allcohort_maytoaug_11wks_cap4to5_data.csv')
