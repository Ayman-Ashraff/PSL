<h1>Back-end</h1>

<p>For all files, please update the paths and run the files in the order that they are mentioned here</p>

<ol>

<li><p>To run the <strong>Augmentation.py</strong> file:</p>
<ul>
<li>It is necessary that WLPSL dataset has been downloaded before running this file</li>
</ul></li>


<li><p>To run <strong>KeypointExtraction.py</strong>:</p>
<ul>
<li>It is necessary that Augmentation.py has been run before running this file</li>
</ul></li>


<li><p>To run <strong>Keypoint_Concat.py</strong> file:</p>
<ul>
<li>It is necessary that KeypointExtraction.py has been run before running this file</li>
</ul></li>


<li><p>To run <strong>psl_model_train.ipynb</strong> file:</p>
<ul>
<li>Use the keypoints and labels files generated by Keypoint_Concat.py as inputs for this file</li>
</ul></li>


<li><p>To run the <strong>TestModel.py</strong> file:</p>
<ul>
<li>Download and save the trained model from psl_model_train.ipynb file</li>
</ul></li>


<li><p>The models folder contains 4 models that our team has trained. All 4 models are capable of recognizing and translating the signs of the following 15 words:<br>
"احتیاط" , خطرناک" ,"بہت پرجوش" ,"دور" ,"مضحکہ خیز" ,"اچھی" ,"صحت مند" ,"بھاری" ,"اہم" ,"ذہین",<br>
"دلچسپ" ,"نہیں", "جلدی" ,"تیار", "جی ہاں"</li>


<li><em>Note:</em>Model architecture can be found in the psl_model_train.ipynb file</li></ul></li><br>


<p>Details on the files:</p>
<ul>

<li><strong>Augmentation.py</strong> is for extracting frames from the videos, performing different operations on them (crop, rescale, skew, combinations), and then saving them in their designated folders</li><br>

<li><strong>KeypointExtraction.py</strong> is for going through all of the classes and their augmentations and generating a keypoints file for each frame which is then saved as an np array file (you might want to run the ShortlistClasses.py script again to make folders and subfolders for storing np array files)</li><br>

<li><strong>Keypoint_Concat.py</strong> file is for going through all the generated np array files and concatenating them into one giant 3D array which can then be fed to the model for training purposes</li><br>

<li>The <strong>.ipynb notebook</strong> is for training PSL models</li><br>

<li><strong>TestModel.py</strong> is for testing the model with new videos to see how it performs in real-world environment</li><br>
</ul>
