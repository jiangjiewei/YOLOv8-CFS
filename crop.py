#!/usr/bin/env python
# coding: utf-8


import os
import xml.etree.ElementTree as ET
from random import randint
from PIL import Image
import shutil
from tqdm import tqdm

data_root="/home/jiangjiewei/dingke_gpu/data_mobile/nocropped_all"  #/test1/keratitis
#des_root=data_root
des_root='/home/jiangjiewei/dingke_gpu/data_mobile/cropped/cropped_all_1024'    #/test/keratitis
label_root='/home/jiangjiewei/peimengjie/external_PHONE_1019/Annotations'

des_dir=['keratitis','normal','other']


for j in range(len(des_dir)):
    data_root_path=os.path.join(data_root,des_dir[j])
    des_root_path=os.path.join(des_root,des_dir[j])


    if not os.path.exists(des_root_path):
        os.mkdir(des_root_path)

    import xml.etree.ElementTree as ET



    size=1024



    xml_files=filter(lambda f:f.endswith(".xml"),os.listdir(label_root))  #8patch_random
    # f=open(os.path.join(data_root,'not_normal.txt'),'w')
    jpgs = filter(lambda f: not f.endswith(".xml"), os.listdir(data_root_path))
    for jpg_name in tqdm(jpgs, desc=f"Processing {des_dir[j]}"):
        jpg_name_ori=jpg_name
        # jpg_name=jpg_name.replace(" ","")
        # jpg_name = jpg_name.replace(" ", "")
        # jpg_name = jpg_name.replace(" ", "")
        # jpg_name = jpg_name.replace(" ", "")
        # jpg_name = jpg_name.replace(" ", "")
        # jpg_name = jpg_name.replace(" ", "")
        # jpg_name = jpg_name.replace(" ", "")
        # jpg_name = jpg_name.replace(" ", "")
        # jpg_name = jpg_name.replace(" ", "")
        if jpg_name.endswith(".jpg") or jpg_name.endswith(".bmp") or jpg_name.endswith(".BMP") or jpg_name.endswith('.JPG'):
            xml_name=jpg_name[:-4]+'.xml'
        else:
            xml_name = jpg_name + '.xml'

        regions=[]
        reg_dict={}
        tree=ET.parse(os.path.join(label_root,xml_name))
        root=tree.getroot()

    #     if not os.path.exists(os.path.join(data_root,des_root,folder_name)):
    #         os.mkdir(os.path.join(data_root,des_root,folder_name)) #单个图片文件切割后保存文件夹
        folder_name=jpg_name_ori[:-4]

        bmp_name = folder_name + ".bmp"
        if os.path.exists(os.path.join(data_root_path,jpg_name_ori)):
            im = Image.open(os.path.join(data_root_path,jpg_name_ori))
        else:
            im = Image.open(os.path.join(data_root_path,bmp_name))
        for ob in root.findall('object'):
            if ob.find('name').text=='conjunctiva':
                loc1=[]
                bndbox=ob.find('bndbox')
                loc1.append(int(bndbox[0].text))
                loc1.append(int(bndbox[1].text))
                loc1.append(int(bndbox[2].text))
                loc1.append(int(bndbox[3].text))
            elif ob.find('name').text=='cornea':
                loc2=[]
                bndbox=ob.find('bndbox')
                loc2.append(int(bndbox[0].text))
                loc2.append(int(bndbox[1].text))
                loc2.append(int(bndbox[2].text))
                loc2.append(int(bndbox[3].text))
        regions1=[]
        regions2=[]
        try:
            x_=randint(max(loc1[0],loc2[0]),min(loc1[2],loc2[2])-size)
            regions1.append((x_,loc1[1],x_+size,loc1[1]+size))
            y_=randint(max(loc1[1],loc2[1]),min(loc1[3],loc2[3])-size)
            regions1.append((loc1[2]-size,y_,loc1[2],y_+size))
            x_=randint(max(loc1[0],loc2[0]),min(loc1[2],loc2[2])-size)
            regions1.append((x_,loc1[3]-size,x_+size,loc1[3]))
            y_=randint(max(loc1[1],loc2[1]),min(loc1[3],loc2[3])-size)
            regions1.append((loc1[0],y_,loc1[0]+size,y_+size))
            reg_dict['conjunctiva']=regions1
            x_=randint(max(loc1[0],loc2[0]),min(loc1[2],loc2[2])-size)
            regions2.append((x_,loc2[1],x_+size,loc2[1]+size))
            y_=randint(max(loc1[1],loc2[1]),min(loc1[3],loc2[3])-size)
            regions2.append((loc2[2]-size,y_,loc2[2],y_+size))
            x_=randint(max(loc1[0],loc2[0]),min(loc1[2],loc2[2])-size)
            regions2.append((x_,loc2[3]-size,x_+size,loc2[3]))
            y_=randint(max(loc1[1],loc2[1]),min(loc1[3],loc2[3])-size)
            regions2.append((loc2[0],y_,loc2[0]+size,y_+size))
            reg_dict['cornea']=regions2
            regions=reg_dict['conjunctiva']+reg_dict['cornea']
        except:
    #         shutil.rmtree(os.path.join(data_root,des_root,folder_name))
    #         f.write(xml_name+'\n')
    #         f.flush()
            size2=128
            regions1=[]
            regions2=[]
            x_=randint(max(loc1[0],loc2[0]),min(loc1[2],loc2[2])-size2)
            regions1.append((x_,loc1[1],x_+size2,loc1[1]+size2))
            #获得结膜区最底部的随机x位置的patch块
            y_=randint(max(loc1[1],loc2[1]),min(loc1[3],loc2[3])-size2)
            #角膜区y轴区域
            regions1.append((loc1[2]-size2,y_,loc1[2],y_+size2))
            #获得角膜区最底部最右区域的patch
            x_=randint(max(loc1[0],loc2[0]),min(loc1[2],loc2[2])-size2)
            regions1.append((x_,loc1[3]-size2,x_+size2,loc1[3]))
            y_=randint(max(loc1[1],loc2[1]),min(loc1[3],loc2[3])-size2)
            regions1.append((loc1[0],y_,loc1[0]+size2,y_+size2))
            x_=randint(max(loc1[0],loc2[0]),min(loc1[2],loc2[2])-size2)
            regions2.append((x_,loc2[1],x_+size2,loc2[1]+size2))
            y_=randint(max(loc1[1],loc2[1]),min(loc1[3],loc2[3])-size2)
            regions2.append((loc2[2]-size2,y_,loc2[2],y_+size2))
            x_=randint(max(loc1[0],loc2[0]),min(loc1[2],loc2[2])-size2)
            regions2.append((x_,loc2[3]-size2,x_+size2,loc2[3]))
            y_=randint(max(loc1[1],loc2[1]),min(loc1[3],loc2[3])-size2)
            regions2.append((loc2[0],y_,loc2[0]+size2,y_+size2))
            regions=regions1+regions2

            for idx in range(len(regions)):
                cropped=im.crop(regions[idx])
                cropped=cropped.resize((512,512))
                cropped.save(os.path.join(des_root_path,folder_name+"_%d.jpg" %idx),quality = 100)
        else:
            for idx in range(len(regions)):
                cropped=im.crop(regions[idx])
                cropped.save(os.path.join(des_root_path,folder_name+"_%d.jpg" %idx),quality = 100)
    # f.close()
    print(des_root_path,'裁剪完成')