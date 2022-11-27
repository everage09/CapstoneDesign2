# 전이학습을 활용한 Image CAPTCHA 해독프로그램 구현
* * *

## 개요
* * *
전이학습을 활용해 Image classification 작업을 하는 모델을 훈련시켜 이를 기반으로 Image CAPTCHA 해독을 시도하는 코드를 작성하였습니다.

## 주제 선정 이유
* * *
CAPTCHA(Completely Automated Public Turing test to tell Computers and Human Apart)는 사람과 컴퓨터를 구분하기 위해 사용되는 방법의 통칭으로, 다양한 방면으로
활용되지만 주로 로그인이나 회원가입 등에 하나의 보안 인증 절차로 사용됩니다. 주로 글자를 읽고 이를 입력하는 Text CAPTCHA와 이미지를 활용한 IMAGE CAPTCHA가 있습니다. Image CAPTCHA의 대표적인 예시로 
Google의 reCAPTCHA가 있습니다. 

현재에도 많은 사설 업체들이 인력으로 CAPTCHA를 유상으로 풀어주는 서비스를 제공하고 있는데, 이를 자동화하는 고성능 프로그램이 나온다면 기존 CAPTCHA는 보안적으로 안전하지 않게 되고,
더욱 강화된 보안 매체를 사용할 것이 요구됩니다. Text CAPTCHA를 대상으로 머신 러닝을 통해 해독을 시도한 케이스는 쉽게 찾을 수 있으나, Image CAPTCHA를 대상으로 한 케이스는 그 수가 많지 않았습니다.
따라서 본 프로젝트에서는 reCAPTCHA를 대상으로 전이학습을 통해 학습된 모델로 특정 클래스들을 대상으로 이미지 캡차를 푸는 모델을 설계하여 Image CAPTCHA도 머신 러닝을 통해 해결이 가능함을 보이고 자동화로부터 안전하지 않음을 보이는 것이 목표입니다.


## 전이학습을 하는 이유
* * *
Google의 reCAPTCHA는 NxN 사이즈의 그리드 형식으로 구성된 이미지 입니다. 저는 이 CAPTCHA의 해독을 그리드 내 각 셀에 대한 Image classification 작업으로 재해석하였고, 이를 수행하는 모델을
만들고자 하였습니다.

<img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/image-recaptcha-test-sample.png?raw=true" width="50%" height="50%" title="reCAPTCHA_sample" alt="sample reCAPTCHA"><br>
4x4 reCAPTCHA의 예

양질의 학습 데이터를 모으는 것은 어려운 일이고, 머신 러닝은 많은 시행착오를 거치기 때문에 짧은 기간에 충분하게 큰 데이터 셋을 마련한 뒤 모델을 뛰어난 성능을 갖도록 훈련시키는 것은 어려운 일입니다.
따라서 처음부터 모델을 구축하는 것보다, 기존에 훈련된 모델을 베이스로 가중치를 초기화하여 학습을 진행하거나 여기서 일부를 고정시켜 특징 추출기로서 활용하는 방법이 더 효율적으로 모델을 구축하는데 좋습니다.

본 프로젝트에서는 reCAPTCHA에서 흔히 등장하는 문제 유형인 교통/탈 것과 관련된 문제에 주로 나타나는 클래스들을 일부 학습시켰고, 다음과 같습니다: 

    'Airplane', 'Ambulance', 'Bicycle', 'Bus', 'Car', 'Fire hydrant',
    'Helicopter', 'Motorcycle', 'Parking meter', 'Stop sign', 'Taxi',
    'Traffic sign', 'Truck'

학습에 사용된 모델은 Imagenet 기반으로 학습된 Resnet152, Resnext101, Vgg16이 있으며, 각 모델을 베이스로 학습시킨 모델의 가중치는 weight/for report 에 있습니다.
학습 결과 상 정확도에 큰 차이가 없었으므로 어느쪽을 선택하여 사용해도 큰 문제가 없습니다.

학습에 사용한 데이터셋은 Google에서 제공하는 Open Images V6 데이터셋을 사용하였고, FiftyOne 라이브러리를 통해 원하는 클래스의 데이터만 선택하여 다운 받았습니다. </br>
[Dataset_downloader.ipynb] 를 사용하여 이미지를 다운 받을 수 있고, 다운받은 이미지 중 일부 학습에 적합하지 않은 이미지는 제거한 뒤 학습을 진행했습니다.

## 프로젝트 진행 순서
* * *
>1. 데이터셋 다운로드 및 정리
>2. 전이학습을 통해 모델 학습
>3. 임의의 reCAPTCHA 생성을 위한 이미지 수집 (Image Scraping / Crawling)
>4. 수집한 이미지로 랜덤하게 reCAPTCHA 생성
>5. reCAPTCHA 이미지를 조각으로 나누어 모델에 통과시켜 예측값을 얻고, 이를 종합하는 해독 프로그램 작성

&#43; data aumentation을 통해 학습 데이터 수를 늘리고 imgae classification 모델을 다시 학습시켜서 정확도 개선을  시도

## 프로젝트 환경
프로젝트를 진행한 환경은 다음과 같습니다.
>1. Python 3.10.6 ver
>2. Pytorch 1.12.1+cu113
>3. CUDA / CUDA toolkit 11.7
>4. cudnn 8.5.0
>5. GPU: RTX3060

소스코드를 실행하는데 필요한 라이브러리는 requirements.txt 파일에 기재되어있고, 이를 통해 설치가 가능합니다. 


## Sample Output:
초록색 테두리는 예측 결과가 Target으로 판단한 셀이고,<br>
노란색 테두리는 Target이 예측 결과 top-3값 안에 들어가는 경우로, Target일 수도 있는 셀입니다. <br>
빨간색 테두리는 Target이 없다고 판단한 경우 입니다.
* * *
### 2X2 크기의 reCAPTCHA 해독 결과 샘플
#### Truck &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Airplane
#### Helicopter |&nbsp;Bus

<img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/2x2/0_reCAPTCHA_merge_10_target_Ambulance_.jpg?raw=true" width="300px" height="300px" title="sample 2x2" alt="2x2_1"> <img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/2x2/0_reCAPTCHA_merge_10_target_Airplane_.jpg?raw=true" width="300px" height="300px" title="sample 2x2" alt="2x2_2"><br/>
Target: Ambulance &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Target: Airplane


<img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/2x2/0_reCAPTCHA_merge_10_target_Fire%20hydrant_.jpg?raw=true" width="300px" height="300px" title="sample 2x2" alt="2x2_3"></img> <img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/2x2/0_reCAPTCHA_merge_10_target_Helicopter_.jpg?raw=true" width="300px" height="300px" title="sample 2x2" alt="2x2_4"></img><br/>
Target: Fire hydrant &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Target: Helicopter

* * *
### 3X3 크기의 reCAPTCHA 해독 결과 샘플
#### Truck &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Parking meter &nbsp;&nbsp; | Ambulance
#### Helicopter | Truck &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Bus
#### Car &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Airplane &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Ambulance
<img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/3x3/0_reCAPTCHA_merge_5_target_Bus_.jpg?raw=true" width="300px" height="300px" title="sample 3x3" alt="3x3_1"></img> <img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/3x3/0_reCAPTCHA_merge_5_target_Helicopter_.jpg?raw=true" width="300px" height="300px" title="sample 3x3" alt="3x3_2"></img></br>
Target: Bus &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Target: Helicopter

<img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/3x3/0_reCAPTCHA_merge_5_target_Stop%20sign_.jpg?raw=true" width="300px" height="300px" title="sample 3x3" alt="3x3_3"></img></br>
Target: Stop sign

* * *
### 4X4 크기의 reCAPTCHA 해독 결과 샘플
#### Traffic sign &nbsp;&nbsp;| Truck | Traffic sign &nbsp;&nbsp;&nbsp;&nbsp;| Ambulance 
#### Fire hydrant &nbsp;| Car &nbsp;&nbsp;&nbsp;| Stop sign &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Fire hydrant
#### Helicopter &nbsp;&nbsp;&nbsp;&nbsp;| Bus &nbsp;&nbsp;| Parking meter | Stop sign
#### Bus &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Bus &nbsp;&nbsp;| Airplane &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Airplane
<img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/4x4/0_reCAPTCHA_merge_100_target_Bus_.jpg?raw=true" width="300px" height="300px" title="sample 4x4" alt="4x4_1"> <img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/4x4/0_reCAPTCHA_merge_100_target_Stop%20sign_.jpg?raw=true" width="300px" height="300px" title="sample 4x4" alt="4x4_2"><br>
Target: Bus &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Target: Stop sign

<img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/result%20sample/4x4/0_reCAPTCHA_merge_100_target_Motorcycle_.jpg?raw=true" width="300px" height="300px" title="sample 4x4" alt="4x4_3"></br>
Target: Motorcycle
* * *

## Algorithm
* * *
    class CaptchaDatatset(Dataset):
    (중략)
    def __getitem__(self, idx):
        # load next image
        image = Image.open(self.files[idx])
        image.resize((self.captcha_size,self.captcha_size))
        np_img = np.array(image)
        image_info = []
        image_info.append(np.array(image))
        image_info.append(self.files[idx])
        
        # 이미지를 구역별로 나눔
        patches = []
        for row in range(self.size):
            for col in range(self.size):
                r= 10+4*(row)+row*self.patch_size
                c= 10+4*(col)+col*self.patch_size
                patch = copy.deepcopy(np_img[r:r+self.patch_size, c:c+self.patch_size])
                patch = Image.fromarray(patch, 'RGB')
                patch = self.transforms(patch) # make PIL image to tensor
                patches.append(patch)
        
        # 정답이 기록된 파일이 존재하는 경우
        if self.label_path is not None:
            # get answer from answer file
            label_info = []
            # parse file name to read answer text file with the same name
            filename = self.files[idx].split('/')[-1].split(".")[0]
            
            with open(f"{self.label_path}/{filename}.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    labels = line.rstrip(',\n').split(",")
                    for elem in labels:
                        label_info.append(elem)
            return label_info, patches, image_info
        else:
            return idx, patches, image_info
Pytorch Dataset 클래스를 상속받아 만든 커스텀 데이터셋 클래스를 통해 디렉토리를 지정하여
데이터셋을 선언하고 데이터를 받아올 수 있습니다. 각 reCAPTCHA 이미지를 미리 셀 단위로 잘라서
텐서로 변환하여 image classification의 모델의 input으로 넣을 수 있도록 합니다.

    def solve(model, dataloader, device, class_names, mode="merge", size=4, captcha_size=928,
        save_dir="../temp/"):
    """
    dataloader를 통해 로드된 모든 CAPTCHA이미지를 푼다.
    model: 예측에 사용할 모델
    dataloader: 로드한 데이터셋 불러올 데이터로더
    device: cpu / gpu
    class_names: 클래스들의 이름 리스트
    mode: 캡챠가 구성된 방식 - merge: 이미지들이 붙여져서 만들어진 방식 / divide: 이미지를 나눠서 만들어진 방식
    size: 그리드 한 줄에 있는 이미지 개수
    captcha_size: 전체 캡챠 이미지의 크기
    save_dir: draw_line 함수에 인자로 전달할 이미지 저장 경로
    """
    # check directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    model.eval()
    corrects = 0
    count = 0
    # compute patch size
    patch_size = int((captcha_size-4*size-16)/size)
    
    # get next data
    for idx, (label_info, patches, image_info) in enumerate(dataloader):
        count += 1
        # randomly choose a target
        target = np.random.choice(class_names, 1)[0]
        # compute prediction on loaded data
        result = predict(model, patches, device, target, class_names, save_dir)
        
        pred = result["prediction"] # 모델에서 예측한 타겟이 있는 셀 인덱스들의 리스트
        top3 = result["top3"] # 모델에서 예측한 값 top-3에 타겟이 있는 셀 인덱스들의 리스트
        
        # make tensor to numpy array
        image = np.squeeze(image_info[0]).numpy()
        # make array into PIL image
        image = Image.fromarray(image)
        img_path = image_info[1][0]
        
        # draw border lines
        draw_line(image, target, pred, top3, size, captcha_size, patch_size, idx+1, img_path, save_dir, show_result=False)

데이터로더를 통해 커스텀 데이터셋으로부터 다음 데이터를 받아오고, 랜덤하게 Target을 정해주고 예측을 실행합니다.
훈련시킨 image classification 모델을 통해 찾고자 하는 Target이 있는 셀의 인덱스를 찾습니다.

모델은 이미지를 입력 받으면 출력으로 입력 이미지가 학습된 각 클래스에 속할 확률을 가진 텐서를 반환합니다.
Top-1 값에 해당하는 클래스가 Target과 같은 경우, 셀에 Target이 존재한다고 예측한 것과 같고, 해당 셀의 인덱스는 pred 리스트에 추가합니다.

Top-3 값에 해당하는 클래스 중 (Top-1값 제외) Target 클래스가 있을 경우, 셀에 Target이 존재할 가능성이 있다고 간주하기로 하고, 
해당 셀의 인덱스는 top3 리스트에 추가합니다. 이 지표는 정답으로서 제시하는 것은 아니지만 사용자들이 보조적으로 참고할 수 있도록 제공하는 지표입니다.

pred 리스트에 있는 인덱스의 셀은 초록 테두리로 표시되고, top3 리스트에 있는 인덱스의 셀은 노란색으로 표시합니다.
두 리스트가 모두 비어있을 경우, CAPTCHA에 찾는 Target이 없다고 간주하여 전체 이미지에 빨간색 테두리로 표시합니다.

에측 결과 이미지는 함수 호출 시 지정한 save_dir에 저장됩니다. Default 값은 "../temp/" 입니다.

예측 결과는 텍스트로도 출력되며, 데이터셋 초기화 때 답안파일이 있는 디렉토리를 제공하였을 시,
예측과 답을 비교하여 전체 예측의 정확도를 측정합니다. Target이 랜덤하게 정해지므로 정확도는 매번 다를 수 있습니다.
* * *
## 정확도 측정 결과
<img src="https://github.com/everage09/CapstoneDesign2/blob/main/images/test_accuracy.png?raw=true" title="Accuracy test" alt="solver_test"></br>
used model based on Resnext101

### 보고서 내용과 달라진 점
보고서 작성 당시 4x4 사이즈 CAPTCHA 테스트 결과 5-10%의 Accuracy가 측정되었으며, 이를 개선하기 위한 노력을 기술하였습니다.
이 때의 낮은 정확도의 원인으로;
1. Image classification의 충분히 높지 않은 정확도.
2. 모델이 NxN개의 셀을 판독하는 동안 Target이 있는 셀을 다 못 찾거나 다른 셀을 오판할 때도 오답이 된다는 점.
3. CAPTCHA solving의 난이도가 셀의 개수에 비례하여 높아지는 점.
을 들어 분석하였고, 이를 극복하기 위해 진행했던 노력들을 기술하였습니다.

+ 그리드 축소
+ Thresholding 및 ROC 커브를 통한 최적 threshold 찾기
+ Data augmentation 및 train parameter 조정
+ Top-k 값을 보조로 활용

위의 4가지 방법에 대한 설명은 보고서에 기재되어 있습니다.
이 중 가장 효과가 컸던 방법은 그리드 축소가 task의 난이도를 직접적으로 낮추기 때문에 가장 효과가 좋았습니다.
여기에 Top-k 방법은 보조적 지표로 활용하기 위해 최종 output에 포함시키도록 하였습니다.

그리드 축소 시행 시 3x3 : 30%, 2x2: 60%에 근접한 정확도를 보였었는데, 보고서 제출 이후, Dataloader가 간혹 
데이터를 Shuffle=False로 하였음에도 순서대로 가져오지 않는 경우가 발생하는 것을 발견했습니다. 
따라서 읽는 CAPTCHA 파일과 답안 파일의 이름을 동일하게 설정하여, 잘못된 파일을 읽는 경우를 방지하였고, 
이후 테스트에서는 위의 표와 같은 정확도가 나오는 것을 발견하였습니다.

그럼에도 불구하고 4x4 크기의 CAPTCHA를 대상으로 한 예측은 80% 아래의 정확도를 보여, 여전히 성능에 개선이 필요합니다.

## 결론 및 발전 가능성
* * *
본 프로젝트를 통해 일정 수준 이상 학습된 Image classification 모델을 통해 그리드 사이즈가 작은 CAPTCHA 이미지의 해독이
가능함을 보일 수 있었다. 단 실제 CAPTCHA는 더 많은 클래스를 포함하고 있고, 그리드의 사이즈가 커질 수록 해독 정확도가 떨어지므로, 
task의 난이도가 상당히 높음을 알 수 있었습니다. 

더 나은 수준의 결과를 위해서는 더 정밀한 분류 모델을 구축할 필요가 있을 것으로 보이며, 이를 위해서는 대용량의 학습데이터와
장시간의 학습이 필요할 것으로 보입니다. 

Image classification을 활용하는 방법 외에는 Object detection을 활용하는 방법이 있을 것으로 보입니다. 
한 셀에 여러 물체가 있을 경우 Image classification의 정확도가 떨어지는 경우가 있었는데, Object detection을 활용하면, Target이
발견된 셀을 찾기만 하면 될 것으로 예상됩니다. 이를 상용중인 object detection service를 활용하여 실험한 사례로 

HOSSEN, Md Imran, et al. An Object Detection based Solver for {Google’s} Image {reCAPTCHA} v2. In: 23rd
international symposium on research in attacks, intrusions and defenses (RAID 2020). 2020. p. 269-284.

이 있습니다. 

## 참고문헌
* * *
1. von Ahn, Luis; Blum, Manuel; Hopper, Nicholas J.; Langford, John (May 2003). "CAPTCHA: Using Hard AI
Problems for Security". Advances in Cryptology — EUROCRYPT 2003. EUROCRYPT 2003: International
Conference on the Theory and Applications of Cryptographic Techniques. Lecture Notes in Computer Science.
Vol. 2656. pp. 294–311.
2. Mitchell, T. M., & Mitchell, T. M. (1997). Machine learning (Vol. 1, No. 9). New York: McGraw-hill.
3. 오일석, (2017). 기계학습 pp. 23-25, 194-200, 225-229.
4. HOSSEN, Md Imran, et al. An Object Detection based Solver for {Google’s} Image {reCAPTCHA} v2. In: 23rd
international symposium on research in attacks, intrusions and defenses (RAID 2020). 2020. p. 269-284.
5. Pytorch reference: Pytorch documentation / tutorial
6. 김건우, 염상준 (2019), (펭귄브로의)3분 딥러닝 파이토치맛, 한빛미디어, ISBN: 9791162242278
93000