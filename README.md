# EMLO 
## Assignment-04: Deployment of Cifar10 dataset

### What's in it?
1. This is CIFAR10 Dataset deployment on gradio
2. This is completely based on 

        - hydra framework
        - Gradio
        - Pytorch Lightning
        - It is also dockeried and pushed in Docker Hub
3. [Docker Hub Link](https://hub.docker.com/repository/docker/vikasran/emlo_assign_04)
4. How to run it:

    - Clone the repo and run the below commands:
        - python3 src/demo_scripted.py
    - from docker:
        - pull the docker imge
            - docker pull vikaran/emlo_assign_04
            -  Then run the commande: **docker run -it vikasran/emlo_assign_04 bash**
            - then run the command **python3 src/demo_scripted.py ckpt_path=/workspace/project/logs/train/runs/2022-11-18_12-20-35/model_script.pt**
