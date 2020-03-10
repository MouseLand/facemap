
def get_frame():
    while 1:
        fsnew = glob.glob(os.path.join(self.folder, 'frame%d.png'%self.irand))
        if len(fsnew)==1:
            self.flag = 0
            self.irand += 1
            fs = fsnew
        else:
            if self.flag==1:
                time.sleep(.01)
            else:
                self.flag = 1
                break
    print('found frame %d'%(self.irand-1))
    self.file_name.setText('frame%d.png'%(self.irand-1))

    frame = cv2.imread(fs[0])

    eye = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (224,224))
    img = np.float32(eye)
    img = normalize99(img)
    self.img.setImage(img*255.)
