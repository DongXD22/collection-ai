import asyncio
from bilibili_api import video
import bilibili_api.video_zone as vz
import requests
import shutil,os
import pandas as pd

idx=0
csv={'path':[],'label':[]}
PATH=r'dataset/images'

async def download_image(url, save_path):
    try:
        folder_path = os.path.dirname(save_path)
        
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"创建文件夹: {folder_path}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"下载成功: {save_path}")

    except Exception as e:
        print(f"下载失败：{e}")

async def get_cover_of_zone(
    tid:int,name:str,total_num:int
)->None:
    data=await vz.get_zone_new_videos(tid,1,total_num)
    videos=data['archives']
    urls=[]
    for video in videos:
        urls.append(video['pic'])
    path=PATH
    for i in range(idx,idx+len(urls)):
        img_path=fr'{path}/{name}_{i+1}.jpg'
        await download_image(urls[i-idx],img_path)
        csv['path'].append(img_path)
        csv['label'].append(name)

def init():
    if os.path.exists(PATH):
        shutil.rmtree(PATH)
    os.makedirs(PATH)

if __name__ =='__main__':
    init()
    tids=[217,1,160]
    names=['animal','anime','life']
    num=50
    for i in range(len(tids)):
        asyncio.run(get_cover_of_zone(tids[i],names[i],num))
        idx+=num
    df=pd.DataFrame(csv)
    df.to_csv(r'dataset/labels.csv',index=False)
    
