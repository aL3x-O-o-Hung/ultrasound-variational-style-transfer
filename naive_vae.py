#from networks import *
from naive_network import *
from load_data import *
import gc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from test import *

def init_unet():
    model=UNet(5,[64,128,256,512,1024],1,'style_transfer')
    inputs=tf.keras.Input(shape=(256,256,1,))
    model(inputs)
    return model

def init_styleunet():
    model=StyleUNet(5,[64,128,256,512,1024],1,'style_transfer')
    inputs=tf.keras.Input(shape=(256,256,1,))
    model(inputs)
    return model

def init_styletransfer():
    model=StyleTransfer(5,[64,128,256,512,1024],1,rec=0,
        #p=[0.05, 0.002, 0, 0, 0],
        p=[0,0,0,0,0.01],
        #s=[0, 0, 0, 0.01, 10],
        s=[0.1,0.002,0.001,0.01,10],
        tv=0,
        name='style_transfer')
    inputs=tf.keras.Input(shape=(256,256,2,))
    model(inputs)
    return model


def init_model():
    model=NaiveVAE(6,[64,128,256,512,1024,2048],4,name='naive_vae',use_resnet=True)
    #model=NaiveVAE_(6,[64,128,256,512,1024,2048],4,name='naive_vae',use_resnet=True)
    inputs=tf.keras.Input(shape=(256,256,1,))
    model(inputs)
    return model

def init_test():
    model=UNet(1,6,[64,128,256,512,1024,1024],1)
    inputs=tf.keras.Input(shape=(256,256,1,))
    model(inputs)
    return model


def train_unet(num,folder):
    folder=folder+'unet/'
    epochs=50
    model=init_unet()
    if num<0:
        model.compile(optimizer=tf.keras.optimizers.Adam(0.005))
    else:
        model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
        model.compile()
    #train_dataset=TrainDataset()
    #data_loader=torch.utils.data.DataLoader(train_dataset,batch_size=320,shuffle=True,num_workers=4)
    loader=DataLoader()
    for epoch in range(num+1,epochs):
        #tf.keras.backend.clear_session()
        #gc.collect()
        #for c,x in enumerate(data_loader):
        while loader.more_to_load():
            tf.keras.backend.clear_session()
            gc.collect()
            x=loader.load_data()
            c,l=loader.check_progress()
            print('epoch:',epoch,str(c)+'/'+str(l))
            model.fit(x,x,epochs=1,batch_size=8)
            del x
        model.save_weights(folder+str(epoch)+'.h5',save_format='h5')


def train_styleunet(num,folder):
    folder=folder+'styleunet/'
    epochs=50
    model=init_styleunet()
    if num<0:
        model.compile(optimizer=tf.keras.optimizers.Adam(0.005))
    else:
        model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
        model.compile()
    #train_dataset=TrainDataset()
    #data_loader=torch.utils.data.DataLoader(train_dataset,batch_size=320,shuffle=True,num_workers=4)
    loader=DataLoader()
    for epoch in range(num+1,epochs):
        #tf.keras.backend.clear_session()
        #gc.collect()
        #for c,x in enumerate(data_loader):
        while loader.more_to_load():
            tf.keras.backend.clear_session()
            gc.collect()
            x=loader.load_data()
            c,l=loader.check_progress()
            print('epoch:',epoch,str(c)+'/'+str(l))
            model.fit(x,x,epochs=1,batch_size=8)
            del x
        model.save_weights(folder+str(epoch)+'.h5',save_format='h5')


def train_styletransfer(num,folder,num1,num2):
    temp_folder=folder
    folder=folder+'styletransfer/'
    epochs=50
    model=init_styletransfer()
    if num<0:
        temp_model=init_unet()
        temp_model.load_weights(temp_folder+'unet/'+str(num1)+'.h5',by_name=True,skip_mismatch=True)
        for layer in model.layers:
            for temp_layer in temp_model.layers:
                if layer.name==temp_layer.name:
                    print(layer.name)
                    layer.set_weights(temp_layer.get_weights())

        temp_model=init_styleunet()
        temp_model.load_weights(temp_folder+'styleunet/'+str(num2)+'.h5',by_name=True,skip_mismatch=True)
        for layer in model.layers:
            for temp_layer in temp_model.layers:
                if layer.name==temp_layer.name:
                    print(layer.name)
                    layer.set_weights(temp_layer.get_weights())
        del temp_model
        model.compile(optimizer=tf.keras.optimizers.Adam(0.005))
    else:
        model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
        model.compile()
    #train_dataset=TrainDataset()
    #data_loader=torch.utils.data.DataLoader(train_dataset,batch_size=320,shuffle=True,num_workers=4)
    for epoch in range(num+1,epochs):
        loader1=DataLoader()
        loader2=DataLoader()
        #tf.keras.backend.clear_session()
        #gc.collect()
        #for c,x in enumerate(data_loader):
        while loader1.more_to_load():
            tf.keras.backend.clear_session()
            gc.collect()
            x=loader1.load_data()
            x_=loader2.load_data()
            if x.shape[0]!=x_.shape[0]:
                break
            c,l=loader1.check_progress()
            x=np.concatenate((x,x_),axis=-1)
            print('epoch:',epoch,str(c)+'/'+str(l))
            model.fit(x,x,epochs=1,batch_size=8)
            del x
            model.save_weights(folder+str(epoch)+'_'+str(c)+'.h5',save_format='h5')

def style_transfer_sample(num,folder,out_folder):
    folder=folder+'styletransfer/'
    model=init_styletransfer()
    model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
    model.compile()
    loader=DataLoader(batch_size=32)
    loader0=DataLoader(batch_size=32)
    im=loader.load_data()
    im0=loader0.load_data()
    for j in range(32):
        cv2.imwrite(out_folder+str(j)+'_content.png',im[j,:,:,0]*255)
        cv2.imwrite(out_folder+str(j)+'_style.png',im0[j,:,:,0]*255)
    im=np.concatenate((im,im0),axis=-1)
    for i in range(5):
        res=model(im,is_training=False)
        for j in range(32):
            cv2.imwrite(out_folder+str(j)+'_'+str(i)+'.png',(res[j,:,:,0]*255).numpy())


def style_transfer_sample_(num,folder,out_folder):
    folder=folder+'styletransfer/'
    model=init_styletransfer()
    model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
    model.compile()
    loader=DataLoader(batch_size=32)
    loader0=DataLoader(batch_size=32)
    im=loader.load_data()
    im0=loader0.load_data()
    temp=[]
    i=-1.0
    while i<=1:
        temp.append(i)
        i+=0.1
    for j in range(32):
        cv2.imwrite(out_folder+str(j)+'_content.png',im[j,:,:,0]*255)
        cv2.imwrite(out_folder+str(j)+'_style.png',im0[j,:,:,0]*255)
    im=np.concatenate((im,im0),axis=-1)
    im=tf.convert_to_tensor(im,tf.float32)
    for i in range(21):
        res=model.sample(im,temp[i],is_training=False)
        for j in range(32):
            cv2.imwrite(out_folder+str(j)+'_'+str(i)+'.png',(res[j,:,:,0]*255).numpy())



def train(num,folder):
    epochs=50
    model=init_model()
    if num<0:
        model.compile(optimizer=tf.keras.optimizers.Adam(0.005))
    else:
        model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
        model.compile()
    #train_dataset=TrainDataset()
    #data_loader=torch.utils.data.DataLoader(train_dataset,batch_size=320,shuffle=True,num_workers=4)
    loader=DataLoader()
    for epoch in range(num+1,epochs):
        #tf.keras.backend.clear_session()
        #gc.collect()
        #for c,x in enumerate(data_loader):
        while loader.more_to_load():
            tf.keras.backend.clear_session()
            gc.collect()
            x=loader.load_data()
            c,l=loader.check_progress()
            print('epoch:',epoch,str(c)+'/'+str(l))
            model.fit(x,x,epochs=1,batch_size=8)
            del x
        model.save_weights(folder+str(epoch)+'.h5',save_format='h5')

def get_res(num,model_folder,out_folder):
    model=init_model()
    model.load_weights(model_folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
    loader=DataLoader(batch_size=32)
    loader0=DataLoader(batch_size=32)
    #im=loader.load_first32()
    im=loader.load_data()
    im0=loader0.load_data()
    for j in range(32):
        cv2.imwrite(out_folder+str(j)+'_content.png',im[j,:,:,0]*255)
        cv2.imwrite(out_folder+str(j)+'_style.png',im0[j,:,:,0]*255)
    im=np.concatenate((im,im0),axis=-1)
    for i in range(5):
        res=model(im,is_training=False)
        for j in range(32):
            cv2.imwrite(out_folder+str(j)+'_'+str(i)+'.png',(res[j,:,:,0]*255).numpy())


def res_for_paper(num,folder):
    folder=folder+'styletransfer/'
    model=init_styletransfer()
    model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
    model.compile()
    for i in range(8):
        for j in range(8):
            im1=cv2.imread('paper_data_/'+str(i)+'.png')
            im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)/255.0
            im1=np.reshape(im1,(1,256,256,1))
            im2=cv2.imread('paper_data_/'+str(j)+'.png')
            im2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)/255.0
            im2=np.reshape(im2,(1,256,256,1))
            im=np.concatenate((im1,im2),axis=-1)
            res=model(im,is_training=False)
            cv2.imwrite('paper_data_/'+str(i)+'_'+str(j)+'.png',(res[0,:,:,0]*255).numpy())


def style_interpolation(style1,style2,num,folder,out_folder):
    folder=folder+'styletransfer/'
    s1=cv2.imread(style1)
    s2=cv2.imread(style2)
    s1=cv2.cvtColor(s1,cv2.COLOR_BGR2GRAY)/255.0
    s2=cv2.cvtColor(s2,cv2.COLOR_BGR2GRAY)/255.0
    s1=np.reshape(s1,(1,256,256,1))
    s2=np.reshape(s2,(1,256,256,1))
    model=init_styletransfer()
    model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
    model.compile()
    b1=model.get_style_representation(s1)
    b2=model.get_style_representation(s2)
    loader=DataLoader(batch_size=32)
    im=loader.load_data()
    for j in range(32):
        cv2.imwrite(out_folder+str(j)+'.png',im[j,:,:,0]*255)
    k=0
    for i in [0.125,0.25,0.375,0.5,0.625,0.75,0.875]:
        b=[]
        for j in range(len(b1)):
            b.append(i*b1[j]+(1-i)*b2[j])
        res=model.given_known_dist(im,b,is_training=False)
        for j in range(32):
            cv2.imwrite(out_folder+str(j)+'_'+str(k)+'.png',(res[j,:,:,0]*255).numpy())
        k+=1


def clear():
    folder='style_transfer/styletransfer/'
    for f in os.listdir(folder):
        if '_' in f:
            os.remove(folder+f)
        elif float(f[:-3])>27:
            os.remove(folder+f)


def latent_space_plot(num,folder):
    folder=folder+'styletransfer/'
    model=init_styletransfer()
    model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
    model.compile()
    for i in range(8):
        im1=cv2.imread('paper_data_/'+str(i)+'.png')
        im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)/255.0
        im1=np.reshape(im1,(1,256,256,1))
        b=model.get_style_representation(im1,is_training=False)
        for j in range(len(b)):
            if j==0:
                z=np.reshape(b[j],(1,-1))
            else:
                z=np.concatenate((z,np.reshape(b[j],(1,-1))),axis=-1)
        if i==0:
            zz=z
        else:
            zz=np.concatenate((zz,z),axis=0)
        print(zz.shape)
    pca=PCA()
    pca.fit(zz)
    colors=cm.rainbow(np.linspace(0,1,8))
    for i in range(8):
        if not os.path.exists('find_places/'+str(i)+'.npy'):
            data=load_latent_test(i)
            s=50
            print('i:',i)
            k=0
            while k*s<data.shape[0]:
                print('k:',k)
                end=(k+1)*s
                if end>data.shape[0]:
                    end=data.shape[0]
                ss=end-k*s
                b=model.get_style_representation(data[k*s:end,:,:,:],is_training=False)
                for j in range(len(b)):
                    if j==0:
                        z=np.reshape(b[j],(ss,-1))
                    else:
                        z=np.concatenate((z,np.reshape(b[j],(ss,-1))),axis=-1)
                t=pca.transform(z)
                if k==0:
                    tt=t
                else:
                    tt=np.concatenate((tt,t),axis=0)
                k+=1
            #t=pca.transform(zz)
            np.save('find_places/'+str(i)+'.npy',tt)
        else:
            tt=np.load('find_places/'+str(i)+'.npy')
        plt.scatter(tt[:,0],tt[:,1],color=colors[i],alpha=0.3,s=5)
    #plt.scatter(t[:,0],t[:,1])
    plt.axis('off')
    plt.savefig('test.png',dpi=800)

def latent_space_label(num,folder):
    folder=folder+'styletransfer/'
    model=init_styletransfer()
    model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
    model.compile()
    for i in range(8):
        im1=cv2.imread('paper_data_/'+str(i)+'.png')
        im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)/255.0
        im1=np.reshape(im1,(1,256,256,1))
        b=model.get_style_representation(im1,is_training=False)
        for j in range(len(b)):
            if j==0:
                z=np.reshape(b[j],(1,-1))
            else:
                z=np.concatenate((z,np.reshape(b[j],(1,-1))),axis=-1)
        if i==0:
            zz=z
        else:
            zz=np.concatenate((zz,z),axis=0)
        print(zz.shape)
    pca=PCA(n_components=2)
    pca.fit(zz)

    im=cv2.imread('naive_res_sample/1_style.png')
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/255.0
    b=model.get_style_representation(np.reshape(im,(1,256,256,1)),is_training=False)
    for j in range(len(b)):
        if j==0:
            z1=np.reshape(b[j],(1,-1))
            z2=np.reshape(b[j],(1,-1))
        elif j==1:
            z1=np.concatenate((z1,-np.ones_like(np.reshape(b[j],(1,-1)))),axis=-1)
            z2=np.concatenate((z2,np.ones_like(np.reshape(b[j],(1,-1)))),axis=-1)
        else:
            z1=np.concatenate((z1,np.reshape(b[j],(1,-1))),axis=-1)
            z2=np.concatenate((z2,np.reshape(b[j],(1,-1))),axis=-1)
    z1=pca.transform(z1)
    z2=pca.transform(z2)
    #plt.scatter(z1[:,0],z1[:,1],color='black',marker='x',s=5)
    #plt.scatter(z2[:,0],z2[:,1],color='black',marker='x',s=5)
    plt.arrow(z1[0,0],z1[0,1],z2[0,0]-z1[0,0],z2[0,1]-z1[0,1],width=0.001,head_width=2,alpha=0.3,label='a')

    im=cv2.imread('paper_data_/7.png')
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/255.0
    b1=model.get_style_representation(np.reshape(im,(1,256,256,1)),is_training=False)
    im=cv2.imread('paper_data_/0.png')
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/255.0
    b2=model.get_style_representation(np.reshape(im,(1,256,256,1)),is_training=False)
    ind=0.125
    for j in range(len(b)):
        if j==0:
            z1=np.reshape(ind*b1[j]+(1-ind)*b2[j],(1,-1))
            z2=np.reshape(ind*b2[j]+(1-ind)*b1[j],(1,-1))
        else:
            z1=np.concatenate((z1,np.reshape(ind*b1[j]+(1-ind)*b2[j],(1,-1))),axis=-1)
            z2=np.concatenate((z2,np.reshape(ind*b2[j]+(1-ind)*b1[j],(1,-1))),axis=-1)
    z1=pca.transform(z1)
    z2=pca.transform(z2)
    #plt.scatter(z1[:,0],z1[:,1],color='black',marker='x',s=5)
    #plt.scatter(z2[:,0],z2[:,1],color='black',marker='x',s=5)
    plt.arrow(z1[0,0],z1[0,1],z2[0,0]-z1[0,0],z2[0,1]-z1[0,1],width=0.001,head_width=2,alpha=0.3,label='b')

    im=cv2.imread('paper_data_/5.png')
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/255.0
    b1=model.get_style_representation(np.reshape(im,(1,256,256,1)),is_training=False)
    im=cv2.imread('paper_data_/2.png')
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/255.0
    b2=model.get_style_representation(np.reshape(im,(1,256,256,1)),is_training=False)
    ind=0.125
    for j in range(len(b)):
        if j==0:
            z1=np.reshape(ind*b1[j]+(1-ind)*b2[j],(1,-1))
            z2=np.reshape(ind*b2[j]+(1-ind)*b1[j],(1,-1))
        else:
            z1=np.concatenate((z1,np.reshape(ind*b1[j]+(1-ind)*b2[j],(1,-1))),axis=-1)
            z2=np.concatenate((z2,np.reshape(ind*b2[j]+(1-ind)*b1[j],(1,-1))),axis=-1)
    z1=pca.transform(z1)
    z2=pca.transform(z2)
    #plt.scatter(z1[:,0],z1[:,1],color='black',marker='x',s=5)
    #plt.scatter(z2[:,0],z2[:,1],color='black',marker='x',s=5)
    plt.arrow(z1[0,0],z1[0,1],z2[0,0]-z1[0,0],z2[0,1]-z1[0,1],width=0.001,head_width=2,alpha=0.3,label='c')

    im=cv2.imread('paper_data_/6.png')
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/255.0
    b1=model.get_style_representation(np.reshape(im,(1,256,256,1)),is_training=False)
    im=cv2.imread('paper_data_/3.png')
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/255.0
    b2=model.get_style_representation(np.reshape(im,(1,256,256,1)),is_training=False)
    ind=0.125
    for j in range(len(b)):
        if j==0:
            z1=np.reshape(ind*b1[j]+(1-ind)*b2[j],(1,-1))
            z2=np.reshape(ind*b2[j]+(1-ind)*b1[j],(1,-1))
        else:
            z1=np.concatenate((z1,np.reshape(ind*b1[j]+(1-ind)*b2[j],(1,-1))),axis=-1)
            z2=np.concatenate((z2,np.reshape(ind*b2[j]+(1-ind)*b1[j],(1,-1))),axis=-1)
    z1=pca.transform(z1)
    z2=pca.transform(z2)
    #plt.scatter(z1[:,0],z1[:,1],color='black',marker='x',s=5)
    #plt.scatter(z2[:,0],z2[:,1],color='black',marker='x',s=5)
    plt.arrow(z1[0,0],z1[0,1],z2[0,0]-z1[0,0],z2[0,1]-z1[0,1],width=0.001,head_width=2,alpha=0.3,label='d')

    colors=cm.rainbow(np.linspace(0,1,8))
    for i in range(8):
        t=np.load('find_places/'+str(i)+'.npy')
        print(i,t.shape)
        plt.scatter(t[:,0],t[:,1],color=colors[i],alpha=0.3,s=5,label=str(i+1))
        #plt.legend(loc='upper left')
    plt.axis('off')
    plt.xlim([-190,220])
    plt.ylim([-150,220])
    plt.savefig('test_.png',dpi=800)

def res_new_im(num,folder,im_path):
    folder=folder+'styletransfer/'
    model=init_styletransfer()
    model.load_weights(folder+str(num)+'.h5',by_name=True,skip_mismatch=True)
    model.compile()
    im1=cv2.imread(im_path)[:,130:375,:]
    im1=cv2.resize(im1,(256,256))
    im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)/255.0
    im1=np.reshape(im1,(1,256,256,1))
    for j in range(8):
        im2=cv2.imread('paper_data_/'+str(j)+'.png')
        im2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)/255.0
        im2=np.reshape(im2,(1,256,256,1))
        im=np.concatenate((im2,im1),axis=-1)
        res=model(im,is_training=False)
        cv2.imwrite('paper_data_/new_image_'+str(j)+'.png',(res[0,:,:,0]*255).numpy())





#train(-1,'naive/')
#get_res(7,'naive/','naive_res/')
#train_unet(-1,'style_transfer/')
#train_styleunet(-1,'style_transfer/')
#style_transfer_sample(27,'style_transfer/','naive_res_/')
#train_styletransfer(26,'style_transfer/',1,1)
#style_transfer_sample_(26,'style_transfer/','naive_res_sample_/')
#style_transfer_sample_(9,'style_transfer/','naive_res_sample_new/')
#train_styletransfer(-1,'style_transfer/',1,1)
#res_for_paper(27,'style_transfer/')
#style_interpolation('paper_data_/3.png','paper_data_/6.png',27,'style_transfer/','naive_res_sample_/')
#clear()
#latent_space_plot(27,'style_transfer/')
#latent_space_label(27,'style_transfer/')
res_new_im(27,'style_transfer/','new_im.bmp')