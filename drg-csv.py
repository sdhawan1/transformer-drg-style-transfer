import boto3
import csv
import numpy as np

"""
# load existing aws urls
# iterate thru generated captions - map to images that are available on AWS
# map those overlapping images to reference idx
# write to batched CSV files
"""
LIMIT = 100
TASK_SIZE = 25
N_CAPTIONS = 5


# get images from original flickr8k captions
first_img_to_idx = {} # 
idx_to_image = {} # 
idx_to_caption = {} # 
last_image = None
with open('captions/reference.txt', newline='') as csvfile:
    ref_reader = csv.DictReader(csvfile, delimiter='\t')
    for idx, row in enumerate(ref_reader):
        image = row['image'].split('#')[0]
        caption = row['caption']
        idx_to_image[idx] = image
        idx_to_caption[idx] = caption
        if image != last_image:
            first_img_to_idx[image] = idx
        last_image = image

# aws get image urls 
AWS_ACCESS_KEY_ID='key_id'
AWS_SECRET_ACCESS_KEY='access_key'
BUCKET_NAME = 'mturk-nlp-drg'

s3 = boto3.resource('s3')
location = boto3.client('s3').get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
url_prefix =  "https://s3-%s.amazonaws.com/%s/%s"

arr_urls = []
for bucket in s3.buckets.all():
    for key in bucket.objects.all():
        url = url_prefix % (location, BUCKET_NAME, key.key)
        arr_urls.append(url)

def make_rows(gen_captions): 
    """create rows by finding the overlap between the images of the
     generated captions and the images found on aws. not all images are on aws 
     because of the tier limit. images on aws are randomly saved"""
    arr_captions = []
    for url in arr_urls: 
        url_img = url.split('/')[-1]
        img_idx = first_img_to_idx.get(url_img, -1) # get idx location of aws img if exists
        if img_idx != -1 and img_idx in gen_captions.keys(): # if aws img idx exists in gen idx
            for i in range(N_CAPTIONS): # get all captions for one image
                # get index of caption - double check images match idx_to_image[image]
                lookup_idx = img_idx + i
                ref_img = idx_to_image[lookup_idx]
                if ref_img != url_img: #if names betweeen reference and aws dont match, dont write
                    continue         
                caption_row = {}
                caption_row['image_url'] = url
                caption_row['ref_caption'] = idx_to_caption[lookup_idx]
                caption_row['gen_caption'] = gen_captions[lookup_idx]
                arr_captions.append(caption_row)
    return arr_captions

def write_rows(arr_captions, fname): 
    n_csvs = LIMIT // TASK_SIZE
    limited_arr = np.array(arr_captions[:LIMIT])
    tasks = np.split(limited_arr, n_csvs)

    for i in range(n_csvs):
        out_name = 'mturk/%s/%s_file_%s.csv' % (fname, fname, i)
        with open(out_name, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['image_url', 'ref_caption','gen_caption'])
            writer.writeheader()
            for caption in tasks[i]:
                writer.writerow(caption)
    return 

def baseline(): 
    # load baseline file
    gen_captions = {}
    with open('captions/baseline.txt', newline='') as gen_reader:
        gen_txt = gen_reader.readlines()

    for idx, line in enumerate(gen_txt):
        line = line.split(' ')
        line = ' '.join(line[1:])
        line = line[:-3]
        line = line.replace('"', "", 10)
        line = line.strip()
        gen_captions[idx] = line

    arr_captions = make_rows(gen_captions)
    write_rows(arr_captions, 'baseline')
    return

def random(): 
    # load random deletion file
    gen_captions = {}
    with open('captions/random_deletion.txt', newline='') as gen_reader:
        gen_txt = gen_reader.readlines()

    for idx, line in enumerate(gen_txt):
        line = line.replace('\n', '')
        line = line.replace('[ deleted ]', '')        
        line = line.replace('"', "", 10)
        line = line.strip()
        gen_captions[idx] = line

    arr_captions = make_rows(gen_captions)
    write_rows(arr_captions, 'random')
    return


def pos(): 
    # load pos deletion file
    gen_captions = {}
    with open('captions/pos_deletion.txt', newline='') as gen_reader:
        gen_txt = gen_reader.readlines()

    for idx, line in enumerate(gen_txt):
        line = line.replace('\n', '')
        line = line.replace('[ deleted ]', '')
        line = line.replace('"', "", 10)
        line = line.strip()
        gen_captions[idx] = line

    arr_captions = make_rows(gen_captions)
    write_rows(arr_captions, 'pos')
    return



def main():
    baseline()
    random()
    pos()

  

if __name__ == "__main__": 
    main()
