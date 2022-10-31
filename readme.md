Magroove Data Science Test
=======================

### Intro
The goal of this test is to access your ability to use data science to tackle a real life issue. Because the nature of the issue is just so complex, we do not expect your solution to be definitive or highly performant - we actually don’t really care about the accuracy of your results at all. What we really care about is your thought process or the way you take an issue, distill it into something solvable by a machine and turn all that into code. As you may imagine at this point, this test will revolve around a machine learning problem - a classification one to be more specific, so let’s get to it.

### The Problem
When uploading an album to digital streaming platforms (DPSs) like Spotify, Apple or so on, it is important that all the metadata that goes with that album to be accurate. Metadata includes basic information like the album name, the year when it was recorded, the artists involved in the tracks and much more. In this exercise we are particularly interested in a specific element of the metadata - language.

Language is an important piece of metadata as it helps the DPSs recommend a given album to listeners that understand that language and will most likely appreciate it. Because of that, DSPs are quite strict with the correct language being assigned when receiving an album.

However, it does occur quite frequently for our users to incorrectly fill the language field in our form when submitting an album, which later leads to conflicts between the given user, our company and the DSPs. The goal of this test then is to work on a solution for a simplified version of such problem.

Together with this description you will find two files - dataset-hindi.csv and dataset-spanish.csv. The names are quite intuitive, each contains metadata from albums we already distribute and that we know their metadata are either written in Spanish or Hindi. Each CSV contains three columns - artist_name, release_name and song_name. Even though the names are rather intuitive, you should pay attention to the fact that the columns artist_name and release_name tend to have repeated entries as the same artist can have several albums and an album can have several songs.

**Your goal then is to create a model to classify future metadata as either written in Spanish or Hindi.**

### The deliverables
We expect to receive from you a single python script (run.py) and a requirements (requirements.txt) file listing all the dependencies to run it. Your script should be written using Python 3.x and it should output a model we could later use for inference in a production-like environment. You can use any packages or libraries you want as long as you don’t use any pre-trained model. You are also forbidden to use third party APIs or services for language inference like AWS Comprehend or GCP Translation. Your single script should contain everything from data cleaning and preprocessing all the way to model training and inference.

### Evaluation and final thoughts

Again, even though accuracy is always a good metric, our primary focus will definitely not be performance (we would be running a Kaggle competition if it was =) ). Our attention will be on the logic you used to try and solve the issue and how it translated into code. So don’t worry if your model performs poorly at the end - we would choose nine out of ten times a poor model performance with a smart approach and attempt to solve the problem in question.

And finally, don’t hesitate to get in touch if you have any questions, doubts or simply need a kick start. Here at Magroove we have a strong culture of helping each other out - so you can definitely count on us - and we promise that any calls for help will not affect your evaluation at all. In the end solving challenging problems is mostly about asking the right questions, so well thought questions are more than welcome. So good luck with your test and keep us in the loop!
