import requests
import json
import os
import time
import multiprocessing
import traceback

FINAL_SAVE_DIR = "../scraped-data-files/"
OCULUS_API = 'https://graph.oculus.com/graphql'

access_token = 'OC|1317831034909742|'  # for oculus store

# doc_id tells oculus which data you want
doc_id_for_app_list = '4145802578837812'
doc_id_for_meta_data = '4282918028433524'
doc_id_for_reviews = '3593832670651594'

section_id = "174868819587665"


def get_all_games():

    full_application_list = []

    payload = {
        'forced_locale': 'en_US',
        'access_token': access_token,
        'doc_id': doc_id_for_app_list,
        'variables': ''
    }

    has_next_page = True
    end_cursor = 'null'

    while has_next_page:

        payload['variables'] = '{{"sectionId" : "{0}", "sortOrder":null, "sectionItemCount":128, "sectionCursor":"{1}"}}'.format(
            section_id, end_cursor)

        response = requests.post(OCULUS_API, data=payload, timeout=30)
        items = response.json()
        
        applications = items['data']['node']['all_items']['edges']

        for item in applications:

            object = {"canonicalName": item['node']['canonicalName'],
                      "appId": item['node']['id'],
                      "displayName": item['node']['display_name']
                      }
            if object not in full_application_list:
                full_application_list.append(object)

        has_next_page = eval(str(items['data']['node']['all_items']
                                 ['page_info']['has_next_page']))

        end_cursor = items['data']['node']['all_items']['page_info']['end_cursor'] if has_next_page else ''

    with open(os.path.join("full_app_list.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(full_application_list, indent=4, sort_keys=True))


    return full_application_list


# get meta data of a game app  
def get_meta(app_id):

    payload = {
        'forced_locale': 'en_US',
        'access_token': access_token,
        'doc_id': doc_id_for_meta_data,
        'variables': ''
    }

    payload['variables'] = '{{"itemId" : "{0}", "first":5,"last":null,"after":null,"before":null,"forward":true,"ordering":null,"ratingScores":null}}'.format(
            app_id)

    response = requests.post(OCULUS_API, data=payload, timeout=30)
    response.raise_for_status()
    items = response.json()

    return items['data']['node']

# get reviews of a game app  
def get_review(app_id):

    payload = {
        'access_token': access_token, 
        'doc_id': doc_id_for_reviews,
        'variables': ''
    }

    full_review_lists = []
    
    has_next_page = True
    end_cursor = ''
    
    while has_next_page:
        payload['variables'] = '{{"id" : "{0}", "first":1024,"last":null,"after":"{1}","before":null,"forward":true,"ordering":"top","ratingScores":[1,2,3,4,5]}}'.format(
            app_id, end_cursor)

        response = requests.post(OCULUS_API, data=payload, timeout=30)
        response.raise_for_status()
        items = response.json()

        full_review_lists.extend(items['data']['node']['firstQualityRatings']['edges'])
        
        has_next_page = False if not items['data']['node']['firstQualityRatings']['page_info']['end_cursor'] else True
        end_cursor = items['data']['node']['firstQualityRatings']['page_info']['end_cursor'] if has_next_page else ''

        time.sleep(1)

    return full_review_lists

# one app_list_item for each process
def process(app_list_item):
    app_id = app_list_item['appId']
    file_name = app_list_item['canonicalName']+ ".json"

    retry = 0
    while retry < 1:
        try:
            meta = get_meta(app_id)
            time.sleep(1)
            reviews = get_review(app_id)
            
            meta['firstQualityRatings']['edges'] = reviews
            
            with open(os.path.join(FINAL_SAVE_DIR, file_name), 'w', encoding='utf-8') as fp:
                fp.write(json.dumps(meta,indent=4, sort_keys=True))
            
            print("Done {0}".format(file_name))
           
            return

        except:
            traceback.print_exc()
            print("Retrying {0}, {1}, {2}".format(file_name, app_id, retry))
            retry += 1
            time.sleep(10)

    print("Giving up {0} {1}".format(file_name, app_id))


def main():

    if not os.path.exists(FINAL_SAVE_DIR):
        os.makedirs(FINAL_SAVE_DIR)

    full_game_list = get_all_games()
    print(len(full_game_list))  

    p = multiprocessing.Pool(multiprocessing.cpu_count()) # number of processes -> cpu cores 
    p.map(process, full_game_list)

        
if __name__ == "__main__":
    main()
