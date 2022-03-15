---
title:  "[Github] Git basic training"

categories:
  - Git
tags:
  - Git

toc: true
toc_sticky: true
 
date: 2022-03-15
last_modified_at: 2022-03-15
---


# Git

git 의 기초 사용 방법

## Git 개념

### Create Repository 

[github.com](github.com) 에서 repository 생성 후 `private,public` 결정하여 생성(이후에 변경 가능)

이곳에서 파일들의 내용을 온라인으로 확인 가능

### Commit

`master` 는 마지막 commit을 가리키고, `head` 는 현재 커밋을 가리킨다

하나의 단위작업을 upload <br>
commit changes에 간단한 설명을 추가

`<>code` 부분에서 commits 들어가면 이 repository 에 대한 history 볼 수 있다

project의 변화를 체계적으로 관리 가능

commit에서 `<>` 버튼을 누르면 이 시점에 추가된 파일들을 확인 가능 : `snapshot`

### checkout

checkout은 head가 가리키는 브랜치를 이동하는 것


### head

head는 현재 working directory가 어떤 버전인지를 가리킨다

head 의 위치를 보고 working directory 를 알 수 있다

### push

### pull

## Git 설정

### Git 설치 확인

[git-scm.com](https://git-scm.com/)

vscode 에서 터미널 실행하여 Git bash 설정이 보이면 설치되어있는거다

터미널에 `git` 실행 시 실행할 수 있는 명령어 목록이 나온다

## vscode

### vscode 설정

git graph 확장 설치

먼저 vscode 에서 version control 할 폴더 연다

`ctrl + shift + g` 눌러 소스 제어 화면에서 리포지토리 초기화 

이를 진행하면 vscode 에서 `/.git` 폴더가 보인다

이 때 `/.git` 폴더가 안보이면 설정에서 `exclude` 검색하여 `/.git` 폴더 가시설정

그러면 `/.git` 폴더가 포함된 디렉토리가 프로젝트 영역이 되고, 이를 `/.git`과 함께 복제하여 보내면 그 사람도 같이 쓸 수 있다

vscode 내에서 이 디렉토리 내에 파일을 수정하면 소스 제어 화면에서 커밋할 수 있다

터미널에 `git log` 시에 history 볼 수 있다. log 종료 시에는 `q`

### stage area

U : Untracked 
M : 


소스 제어 화면에서 변경 사항의 파일 옆의 `+` 를 누르면 스테이징된 변경사항으로 이동하고, 이는 커밋할 파일을 담는 장바구니 역할을 한다

이를 통해 version control 시에 단위작업들을 나누어 줄 수 있다

### backup

### code 가져오기

https 주소를 가져와 소스 제어 화면에서 원격 추가 하여 이름은 origin으로 설정<br>
원격 저장소를 내 로컬 저장소로 가져온다

소스 제어 화면에서 분기 게시 시 `git graph` 에서 `master` 옆에 `origin` 추가되고 원격 repository 에 내 로컬 저장소의 파일이 추가된다

`git graph`의 `origin` 마크는 그 버전 까지 원격 저장소에 올라갔다는 뜻이고, master는 마지막 버전을 가리킨다

그 다음 부터는 커밋 시 `master(branch)` 마크만 뜨고 소스 제어 화면에 변경내용 동기화 버튼이 생긴다 == `pull`

**pull = fetch + merge**

### vscode 로 원격 저장소에서 로컬 저장소로 복제해오기

원격 저장소의 code 버튼을 눌러 https url을 복사해 vscode 에서 새 창을 연다.<br>
소스 제어 화면에서 리포지토리 복제 버튼을 눌러 주소를 누르고 가져올 폴더를 정해주면 그 위치로 복제된다

## version control

Git 은 원격 저장소와 로컬 저장소의 버전이 따로 존재 한다 

a,b,원격 저장소가 있을 때 a 사용자가 원격 저장소에 push를 하면, 원격 저장소에 해당 version이 올라가고 b 사용자가 로컬 저장소로 push 를 하면 같은 버전을 로컬에 받아올 수 있다

origin/head 는 원격 저장소의 정보가 어디를 가리키는지의 정보


### 원격 repo 업데이트

원격 repo 를 가져온 폴더를 vscode 에서 열었을 때 만약 원격 repository 에 update 된 version 이 있다면, `pull` 을 통해 업데이트된 내용을 가져올 수 있다

master 표시와 origin/master 표시가 다른것은 원격 저장소가 다른 내용이 있다는 내용인데, 아직 로컬 저장소에 적용을 하지 않은 상태라는 것을 나타낸다

만약 a,b 사용자가 있을 때 a가 먼저 원격 저장소에 push 하고 b사용자가 push 하려고 하면, 먼저 pull 을 실행하라고 나온다

이때는 merge commit 하면 된다

b 사용자가 push 하기 전 먼저 pull 하고 그 다음 push 를 하면 merge 된 branch 가 생긴다

![merge](/assets/images/git-practice/merge.png)

### 병합 충돌

같은  파일을 서로 다른 사용자가 다르게 수정했을 때 나타난다

이때는 먼저 push 한 사람의 코드가 기준이 된다

바뀐 쪽의 code 에서 변경점을 보여주는데, `<<<<<HEAD` 부분은 현재 사용자의 변경점이고, `>>>>>!~~` 는 먼저 push 한 사람의 코드이다

이때 변경점에 대해 여러 방법이 존재한다

accept current change: 내 변경점을 쓰겠다<br>
accept incoming change: 상대방의 변경점을 쓰겠다<br>
accept both changes: 양쪽 다의 변경점을 둘다 가져온다<br>
compare changes: 두 변경점을 비교한다

만약 .orig 라는 파일이 생기면 이 파일은 백업 파일이다

#### 3-way merge

1. a,b 사용자가 있을 때 a 사용자가 먼저 같은 파일의 변경점을 커밋하고 push 한다.<br>
2. b 사용자는 먼저 pull 하여 파일을 받은 뒤 push 를 해야하는데, pull 했을 때 병합 충돌이 일어나게 된다.<br>
3. 위에 있는 방법으로 b 사용자가 병합 충돌을 해결하고, 커밋한 뒤 push 하면 a 사용자가 pull로 파일을 받아온다

### version 롤백

`reset`,`revert` 로 롤백 가능하다

### 새로운 version 생성 시

새로운 version 이 생성되면 head가 가리키는 version이 parent가 되고, head는 마스터를 가르킨다. <br>
이때 master가 새로운 version을 가리키게 된다

### 이전 version 선택 시

`git checkout *version*`

이 코드를 실행하면 master는 가만히 있고 head가 이전 version을 가리킨다

### Detatched state

head가 master 가 아닌 version을 가리킬 때를 말한다

이때 새로운 version을 만들면, master 가 가리키는 version이 바뀌는게 아니라 head가 가리키는 version이 변하게 된다

### Branch

#### Remote tarcking branch 

커밋 시 master 밑에 origin/master branch가 하나 생기는데, 이를 `remote tracking branch` 라고 한다
