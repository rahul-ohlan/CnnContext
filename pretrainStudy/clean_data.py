import os
import shutil
import re
# move to stimuli_directory
target_path = r"C:\Users\rahul__ohlan\OneDrive\Desktop\testFolder\pretrainStudy\data\STIMULI-Shira-F73"
os.chdir(r"C:\Users\rahul__ohlan\OneDrive\Desktop\testFolder\pretrainStudy\data\STIMULI-Shira-F73")
print(os.listdir())

import shutil
import re
pattern = re.compile(r"\((\d)\)")

for directory in os.listdir():
    path = target_path+"\\"+directory
    if not path.endswith(".DS_Store"):
        if not os.path.isdir(path):
            os.remove(path)

    # now only directories remain
    # now clean all contexts inside the directory
        print("moving to: ", path)
        os.chdir(path)
        # remove "scenes" folders
        inside_contexts = os.listdir(os.getcwd())
        if "scene" in inside_contexts or "Scenes" in inside_contexts or "scenes" in inside_contexts:
            print("deleting scenes ...\n")
            try:
                shutil.rmtree("Scenes")
            except:
                shutil.rmtree("scene")
            print("done!")
        # remove .m files
        print ("cleaning directory...")
        for image in inside_contexts:
            if image.endswith(".m"):
                os.remove(path+"\\"+image)
            file_number = pattern.findall(image)
            if file_number and int(file_number[0]) > 5:
                print("removing file: ", path+"\\"+image)
                os.remove(path+"\\"+image)
        print("moving back to contexts_directory")
        os.chdir("../")
    else:
        try:
            os.remove(path)
        except:
            continue