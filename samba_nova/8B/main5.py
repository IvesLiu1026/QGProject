import os
import openai
import json
import re
import time
from dotenv import load_dotenv
import sys

page_number = sys.argv[1]

load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv("YU_XIANG_SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

DATASET_DIR = "../dataset/split_dataset"
TEST_OUTPUT_FILE = "../results/purpose_driven_test_results.jsonl"

dataset_file = f"{DATASET_DIR}/dataset_page_{page_number}.jsonl"

# Load dataset
with open(dataset_file, "r", encoding="utf-8-sig") as file:
    data = [json.loads(line) for line in file]

# Extract titles, categories, transcripts
titles = [item["title"] for item in data]
categories = [item["category"] for item in data]
transcripts = [item["transcript"]["en"] for item in data]


simple_summary_prompt = """
You are an expert educational content creator. 
Based on the following video transcript, 
first, provide a comprehensive summary that captures 
the main points, 
key facts, 
and important details. \
    
Ensure the summary is detailed and objective. 
Then, generate ten multiple-choice questions from this summary 
that cover key facts and details. 
Ensure each question has four answer choices and one correct answer.

Provide the summary followed by the questions and answers in the following format:
Summary:
[Your summary here]

1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Transcript:
{transcript}
"""

detailed_summary_prompt = """
You are an expert educational content creator. Based on the following video transcript, first, provide a detailed summary that captures the main points, key facts, and important details. Ensure the summary is comprehensive and covers all significant aspects of the content.

Step 1: Identify Main Ideas and Key Points
- Extract the main ideas from each section of the transcript.
- Identify key facts, important details, and significant findings.

Step 2: Structure the Summary
- Organize the summary into an introduction, body, and conclusion.
- Ensure each part covers different aspects of the content.

Step 3: Maintain an Objective Tone
- Summarize the content objectively without personal bias.
- Use neutral language to convey the information.

Step 4: Ensure Comprehensive Coverage
- Include essential aspects such as context, methodology, results, and conclusions.
- Avoid omitting critical information.

Step 5: Ensure Logical Flow
- Use transition words and phrases to connect ideas.
- Ensure the summary flows logically from one point to the next.

Once the summary is completed, generate ten multiple-choice questions that cover key facts and details from the transcript. Ensure each question has four answer choices and one correct answer.

Provide the summary followed by the questions and answers in the following format:
Summary:
[Your summary here]

1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Transcript:
{transcript}
"""

zero_shot_prompt = """
You are an expert educational content creator. Based on the following transcript, generate ten multiple-choice questions that cover key facts and details. Ensure each question has one correct answer and three distrators.

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Transcript:
{transcript}
"""

one_shot_prompt = """
You are an expert educational content creator. Based on the following example and the provided transcript, generate ten multiple-choice questions.

Example:
transcipt: welcome to kids academy [Music] hello let's get started on the lesson native americans are the original\npeople who lived in the united states native americans lived in tribes and nations\nthat had their own religions traditions and languages [Music] among the famous indian\nnations there are cherokee [Music] sioux [Music] navajo [Music] iroquois [Music] and apache [Music] today there are more\nthan 500 native american tribes and nations let's take a look at the worksheet [Music]\ncan you find the names of five indian tribes or nations in the word search we're looking for\ncherokee sioux navajo [Music] iroquois and apache the words are written horizontally and vertically\nfirst let's find the name cherokee it begins with the letter c cherokee means people with another\nlanguage do you see it in the word search there it is c h e r o k e e awesome job now let's look for sue native americans of the\nsioux tribe were known as warriors and hunters can you find this name it's spelled s i o u x [Music] bounded s i o u x great work the next name\nis navajo it begins with the letter n the navajo tribe originally lived in the\nsouthwest region of the united states let's find the tribal name\nin the word search [Music] n a v a j o correct native americans of the\niroquois tribe lived in the northeast region they were known for fishing and agriculture\nagriculture is another word for farming do you see the name iroquois\nit starts with the letter i [Music] it's right here i r o q u o i s good\njob the last name that we need to find is apache the apache indians built shelters\ncalled wigwams from branches leaves and grass let's find\nthe name apache [Music] here a p a c h e excellent let's review [Music] native americans are\nthe original people who lived in the united states the five largest tribes and nations\nare cherokee sioux navajo iroquois and apache each tribe has its own unique culture\nand traditions they were known for unique skills such as hunting and fishing [Music]\nagriculture is another word for farming a wigwam is a native american shelter\nbuilt from branches leaves and grass thanks for watching goodbye [Music] subscribe\nto our channel to stay updated on new videos find links to our apps in the comments below

1) What does it mean to be a Native American?
    - A: That you were among the largest group of people living in the United States of America.
    - B: That you are among the most important groups of people living in the United States of America.
    - C: That you were one of the original people who lived in the United States of America.
    - D: That you were one of the founding fathers of the United States of America.

2) There are more than 500 Native American tribes and nations. Which 5 tribes were the largest?
    - A: Apache, Cherokee, Iroquois, Navajo, and Sioux
    - B: Cherokee, Anasazi, Chippewa, Pawnee, and Osage
    - C: Apache, Iroquois, Chippewa, Pawnee, and Lakota
    - D: Navajo, Sioux, Lakota, Pawnee, and Osage

3) What does the tribe name Cherokee mean?
    - A: People known as farmers.
    - B: People known as warriors.
    - C: People with another religion.
    - D: People with another language.

4) What is a "wigwam"?
    - A: A farming technique that increases crops.
    - B: A shelter built from branches, leaves, and grass.
    - C: A shelter built from mud bricks and clay.
    - D: A fishing technique that used a spear.


Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Transcript:
{transcript}
"""

few_shot_prompt = """
You are an expert educational content creator. Based on the following example and the provided transcript, generate "five multiple-choice questions".

Example 1:
transcipt: welcome to kids academy [Music] hello let's get started on the lesson native americans are the original\npeople who lived in the united states native americans lived in tribes and nations\nthat had their own religions traditions and languages [Music] among the famous indian\nnations there are cherokee [Music] sioux [Music] navajo [Music] iroquois [Music] and apache [Music] today there are more\nthan 500 native american tribes and nations let's take a look at the worksheet [Music]\ncan you find the names of five indian tribes or nations in the word search we're looking for\ncherokee sioux navajo [Music] iroquois and apache the words are written horizontally and vertically\nfirst let's find the name cherokee it begins with the letter c cherokee means people with another\nlanguage do you see it in the word search there it is c h e r o k e e awesome job now let's look for sue native americans of the\nsioux tribe were known as warriors and hunters can you find this name it's spelled s i o u x [Music] bounded s i o u x great work the next name\nis navajo it begins with the letter n the navajo tribe originally lived in the\nsouthwest region of the united states let's find the tribal name\nin the word search [Music] n a v a j o correct native americans of the\niroquois tribe lived in the northeast region they were known for fishing and agriculture\nagriculture is another word for farming do you see the name iroquois\nit starts with the letter i [Music] it's right here i r o q u o i s good\njob the last name that we need to find is apache the apache indians built shelters\ncalled wigwams from branches leaves and grass let's find\nthe name apache [Music] here a p a c h e excellent let's review [Music] native americans are\nthe original people who lived in the united states the five largest tribes and nations\nare cherokee sioux navajo iroquois and apache each tribe has its own unique culture\nand traditions they were known for unique skills such as hunting and fishing [Music]\nagriculture is another word for farming a wigwam is a native american shelter\nbuilt from branches leaves and grass thanks for watching goodbye [Music] subscribe\nto our channel to stay updated on new videos find links to our apps in the comments below

1) What does it mean to be a Native American?
    - A: That you were among the largest group of people living in the United States of America.
    - B: That you are among the most important groups of people living in the United States of America.
    - C: That you were one of the original people who lived in the United States of America.
    - D: That you were one of the founding fathers of the United States of America.

2) There are more than 500 Native American tribes and nations. Which 5 tribes were the largest?
    - A: Apache, Cherokee, Iroquois, Navajo, and Sioux
    - B: Cherokee, Anasazi, Chippewa, Pawnee, and Osage
    - C: Apache, Iroquois, Chippewa, Pawnee, and Lakota
    - D: Navajo, Sioux, Lakota, Pawnee, and Osage

3) What does the tribe name Cherokee mean?
    - A: People known as farmers.
    - B: People known as warriors.
    - C: People with another religion.
    - D: People with another language.

4) What is a "wigwam"?
    - A: A farming technique that increases crops.
    - B: A shelter built from branches, leaves, and grass.
    - C: A shelter built from mud bricks and clay.
    - D: A fishing technique that used a spear.

5) What is the significance of the Navajo tribe?
    - A: They were known for their skills in agriculture.
    - B: They were known for their skills in hunting.
    - C: They were known for their skills in fishing.
    - D: They were known for their skills in warfare.

Example 2:
transcript: Human history is intertwined with some of Earth's mightiest species, but there is one graceful creature that rises above the\nrest. Only when seen it in its ideal habitat, can\nwe truly appreciate this king among birds. Behold: the nobel pigeon.\" If you think they're only good for pooping\non statues, then think again. [OPENING MUSIC] In On the Origin of Species, Charles Darwin\npresented an idea that changed the world. He knew if he was right, this idea was gonna\nturn science on its head, so in chapter 1, you know what he chose as his very first\nexample? It wasn't the tortoises, or finches, or\neven the giant fossil armadillos he found during his journeys. He chose pigeons. Over giant fossil armadillos. But he had a good reason, and if you think\notherwise, then you've never seen FANCY pigeons. These are the birds that got Darwin's attention,\nbecause underneath all that feathery fashion, is just one species, like dogs with wings\ninstead of rats with wings. All of that variation was tweaked from one\nancient mold. The wild rock dove. Thousands of years before they were eating\nold hot dog buns out of the trash, these birds were found on seaside cliffs, but pretty much\nas soon as cities sprung up, they moved in, because to a pigeon, a building is just\na cliff with better architecture. Pigeons are uniquely suited to city life,\nbut they were only able to conquer all of Earth's urban areas because we brought them\nthere. Why? Because we liked feeding them… to ourselves. In fact, from Egypt to Rome to the early 20th\ncentury, the main roles of a pigeon were dinner, or something for rich people to breed into\nsilly shapes. Over time here and there, a few of these domestic\nbirds escaped and returned to a \"wild\" life, they just never left the city. But for some reason along that journey, our opinion\nof pigeons went from this… to this. Watch pigeons pecking at the sidewalk and\nyou're not looking at the smartest birds in the tree. They can't solve puzzles like crows. They can't talk like parrots. A pigeon's brain is only about the size\nof that fingertip… but like most things in nature there's more to the story. Most of their skull is eyeball: if they were\nthe size of humans, their eyes would be as big as grapefruits… Those huge eyes have five color receptors,\ncompared to our three, letting them see things we can't imagine. One pigeon named Linus was trained to remember\nnearly a thousand images. Pigeons can peck out a Monet from a Picasso,\nthey can even judge if a child's drawing is good or bad. Pigeons can tell correctly spelled words among\nmisspelled words, As if they aren't annoying enough.. They can even put groups of objects in numerical\norder, which sounds easy because we're smart, but pigeons do numbers as well as monkeys\ndo, Pigeon vision is the bomb. Literally. During WW II, psychologist B.F. Skinner tried\nto turn these birds into weapons. He trained pigeons to keep an image centered\non a tiny screen by pecking at it. He hooked this up to a navigation system,\nand then loaded it inside of a bomb. He wanted to create explosive missiles piloted\nby kamikaze pigeons. He built several successful prototypes using\nmoney from the General Mills cereal company… yes, the people who make Cheerios, but the\nArmy never let it get off the ground. Pigeon navigation goes a lot farther than\nbird bombs. Just like /human/ city-dwellers, pigeons are\ncommuters, flying out in the morning to find food and coming back at night. They're tightly bonded to their home, and\nthis instinct is so strong that we've used them as messengers for centuries, like Flapchat\ninstead of Snapchat. Before Paul Reuter founded a global news service,\nhe used pigeons to deliver news. During World War I and II, racing pigeons\nwith names like Cher Ami and GI Joe were given actual medals for delivering messages under\nfire. How are pigeons so good at finding their way\nhome from places they've never been? Different experiments have found pigeons use\nvisual maps, Earth's magnetic field, the angle of the sun, even smells to navigate. But when scientists knock each of these senses\nout, some birds can still find their way home. What we do know is pigeons use a lot of senses,\nmaybe even some we don't know about yet. Even though pigeons are everywhere, there's\none thing you never seem to see: Baby pigeons. They do exist, and… they are ugly. But it's a reminder that even a bird that's\neverywhere only gives us glimpses into its life. Darwin's ideas about natural selection were\nborn on a voyage around the world. But you don't have to go to exotic places\nto have your mind blown by evolution's awesomeness. Darwin knew that, and that's why he picked\nthe pigeon to introduce the world to his theory. If you know where to look, wildlife is everywhere that\nwe are: just make sure if you look up to admire it, you keep your mouth closed. Stay curious. I want to thank our friends from BBC Earth\nfor helping us make this episode, because pigeons look awesome in slow-motion. Some of pigeons' oldest enemies have followed\nthem to cities: birds of prey. These scientists are studying peregrine falcons\nfor Planet Earth II, the sequel to the groundbreaking BBC series. It's part of \"Cities\", an entire episode\ndedicated to urban wildlife. Of course, birds of prey aren't pigeons'\nonly urban predators. For Planet Earth II, the team filmed a pigeon\nhunt you have to see to believe. You can find Planet Earth II on BBC One in\nthe UK and coming soon to BBC America. For more information, check out their website.

1) Their eyeballs are a major component of their skulls. On the video, they gave a comparison if humans had pigeon-like eyeballs. How big would the human eyeballs be?
    - A: Human eyeballs would be the size of tomatoes.
    - B: Human eyeballs would be the size of watermelons.
    - C: Human eyeballs would be the size of grapefruits.
    - D: Human eyeballs would be the size of baseballs.

2) Why was Charles Darwin, the father of evolution theory, so interested in pigeons to support his theory?
    - A: Darwin knew that every country had a form of pigeons.
    - B: Darwin knew that despite the differences, pigeons all came from the Rock Dove.
    - C: Darwin was able to examine pigeons more easily than other animals.
    - D: Darwin thought the fancy pigeons would attract more attention than armadillos.

3) What is the size of a pigeon's brain?
    - A: The size of your thumb
    - B: The size of a fingertip
    - C: The size of a walnut
    - D: The size of a golf ball

4) What do pigeons' "super eyes" allow them to do?
    - A: It allows them to be more balanced and agile than humans.
    - B: It allows them to magnify their focus on objects.
    - C: It allows them to remember things humans can't.
    - D: It allows them to see things humans can't.

5) Why did B.F. Skinner try and train pigeons during World War II?
    - A: He trained them to help crash enemy planes.
    - B: He trained them to deliver messages.
    - C: He trained them in order to make them aerial bombers.
    - D: He trained them to crack enemy messages.

Example 3:
transcipt: As your morning alarm blares,\nyou mutter to yourself, \"Why did I set it so early?\" While brushing your teeth,\nyou think, \"I need a haircut... unless?\" Rushing out the front door,\nyou reach for your keys and realize they're not there. Frustrated you shout,\n\"I can't do anything right!\" just in time to notice your neighbor. Being caught talking to yourself\ncan feel embarrassing, and some people even stigmatize this\nbehavior as a sign of mental instability. But decades of psychology research show\nthat talking to yourself is completely normal. In fact, most, if not all, of us engage\nin some form of self-talk every single day. So why do we talk to ourselves? And does what we say matter? Self-talk refers to the narration\ninside your head, sometimes called inner speech. It differs from mental imagery\nor recalling facts and figures. Specifically, psychologists\ndefine self-talk as verbalized thoughts directed toward\nyourself or some facet of your life. This includes personal conversations like\n\"I need to work on my free throw.\" But it also includes reflections\nyou have throughout the day, like \"The gym is crowded tonight.\nI'll come back tomorrow.\" And while most self-talk\nin adults tends to be silent, speaking to yourself out loud\nalso falls into this category. In fact, psychologists believe our first\nexperiences with self-talk are mostly vocal, as children often speak to themselves\nout loud as they play. In the 1930s, Russian psychologist\nLev Vygotsky hypothesized that this kind of speech was\nactually key to development. By repeating conversations\nthey've had with adults, children practice managing their behaviors\nand emotions on their own. Then, as they grow older, this outward\nself-talk tends to become internalized, morphing into a private inner dialogue. We know this internal self-talk\nis important, and can help you plan,\nwork through difficult situations, and even motivate you throughout the day. But studying self-talk can be difficult. It relies on study subjects clearly\ntracking a behavior that's spontaneous and often done without conscious control. For this reason, scientists are still\nworking to answer basic questions, like, why do some people\nself-talk more than others? What areas of the brain are activated\nduring self-talk? And how does this activation differ\nfrom normal conversation? One thing we know for certain, however, is that what you say in these\nconversations can have real impacts on your attitude and performance. Engaging in self-talk\nthat's instructional or motivational has been shown to increase focus,\nboost self-esteem, and help tackle everyday tasks. For example, one study\nof collegiate tennis players found that incorporating instructional\nself-talk into practice increased their concentration\nand accuracy. And just as chatting to a friend\ncan help decrease stress, speaking directly to yourself may also\nhelp you regulate your emotions. Distanced self-talk is when\nyou talk to yourself, as if in conversation with another person. So, rather than\n\"I'm going to crush this exam,\" you might think,\n\"Caleb, you are prepared for this test!\" One study found that this kind\nof self-talk was especially beneficial for reducing stress\nwhen engaging in anxiety-inducing tasks, such as meeting new people\nor public speaking. But where positive self-talk can help you,\nnegative self-talk can harm you. Most people are critical\nof themselves occasionally, but when this behavior gets too frequent\nor excessively negative, it can become toxic. High levels of negative self-talk\nare often predictive of anxiety in children and adults. And those who constantly blame themselves\nfor their problems and ruminate on those situations typically experience\nmore intense feelings of depression. Today, there's a field\nof psychological treatment called cognitive behavioral therapy,\nor CBT, which is partially focused on regulating\nthe tone of self-talk. Cognitive behavioral therapists\noften teach strategies to identify cycles of negative thoughts and replace them with neutral\nor more compassionate reflections. Over time, these tools can improve\none's mental health. So the next time you find yourself\nchatting with yourself, remember to be kind. That inner voice is a partner you'll be\ntalking to for many years to come.

1) One study found that ____ percent of news coverage about Islam and Muslims is negative.
    - A: 20
    - B: 50
    - C: 80
    - D: 100

2) What is one thing going to mosques more frequently linked to?
    - A: More tolerant views of people of other faiths
    - B: Greater civic engagement
    - C: Both A and B
    - D: None of the above

3) ISIS has as much to do with Islam as the Ku Klux Klan has to do with ______.
    - A: The white race
    - B: Christianity
    - C: White supremacy
    - D: Violence

4) According to some psychologists, our first experiences with self-talk are often ___.
    - A: Vocal
    - B: Silent
    - C: Negative
    - D: Confusing

5) Which of the following is NOT described as a function of self-talk?
    - A: Planning
    - B: Motivation
    - C: Emotional regulation
    - D: Perception of time

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Ensure generating five multiple-choice questions.

Category: {category}

Transcript:
{transcript}
"""

purpose_driven_prompt = """
You are an expert educational content creator. Based on the following transcript and the purpose of the questions, generate ten multiple-choice questions and including one correct answer and three distractors.

Types of Questions:
1. Factual Questions: Test recall of specific facts.
2. Conceptual Questions: Assess understanding of concepts.
3. Application Questions: Test ability to apply knowledge to new situations.
4. Analytical Questions: Require analysis and interpretation of information.

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Transcript:
{transcript}
"""

bloom_taxonomy_prompt = """
You are an expert educational content creator. Based on the following transcript, generate ten multiple-choice questions that span across Bloom's Taxonomy levels. Each question should assess one of the following cognitive levels:
You don't have to include the levels in the questions, but ensure that the questions cover a range of cognitive skills.

1. **Knowledge**: Questions that test recall of specific facts from the transcript.
2. **Comprehension**: Questions that check understanding by asking learners to summarize or explain.
3. **Application**: Questions that test the ability to apply knowledge in real-world scenarios.
4. **Analysis**: Questions that require breaking down information into parts and understanding relationships.
5. **Synthesis**: Questions that encourage combining ideas to form new insights or solutions.
6. **Evaluation**: Questions that ask for judgment based on criteria or justification of opinions.

Please provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Correct answer]: [Correct option]

Category: {category}

Transcript:
{transcript}
"""

refinement_prompt = """
You are an expert educational content creator. Based on the following transcript and the initial set of multiple-choice questions, please review and refine the questions to ensure they are:

- Clear and unambiguous.
- Factually accurate and aligned with the transcript content.
- Appropriately challenging for the intended audience.
- Covering key concepts and important details from the transcript.
- Each question has one correct answer and three plausible distractors.

Instructions:

- Correct any factual errors in the questions or answer choices.
- Improve the wording for clarity and conciseness.
- Ensure the distractors are plausible but clearly incorrect.
- Remove any ambiguity in the questions or answer choices.
- Maintain the original format.

Provide the refined questions and answers in the following format:

1) [Refined question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Correct answer]: [Correct option]

Transcript:
{transcript}

Initial Questions:
{initial_questions}
"""


# Different prompt templates list
prompts = [
    # ("zero_shot", zero_shot_prompt),
    # ("purpose_driven", purpose_driven_prompt),
    # ("bloom_taxonomy", bloom_taxonomy_prompt),
    # ("one_shot", one_shot_prompt),
    # ("detailed_summary", detailed_summary_prompt),
    # ("simple_summary", simple_summary_prompt),
    ("few_shot", few_shot_prompt),
    # ("refinement", refinement_prompt)
]

OUTPUT_FILE = "../results/1106/few_shot/page_" + page_number + ".jsonl"


# Define constants for retry mechanism based on rate limits
MAX_RETRIES = 10  # Max number of retries before giving up
INITIAL_BACKOFF = 6  # Start with at least 6 seconds to stay within rate limits
BACKOFF_MULTIPLIER = 1.5  # Increase wait time by this factor on each retry

def generate_questions(prompt, category, transcript):
    retries = 0
    backoff = INITIAL_BACKOFF
    
    while retries < MAX_RETRIES:
        try:
            # Make the API request
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-405B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert educational content creator. "
                            "Your task is to generate comprehensive, multiple-choice questions based on a provided transcript."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt.format(category=category, transcript=transcript)
                    }
                ],
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        
        except openai.RateLimitError as e:
            print(f"Rate limit error encountered: {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)  # Wait before retrying
            retries += 1
            backoff = min(backoff * BACKOFF_MULTIPLIER, 60)  # Cap backoff at 60 seconds

    print("Max retries reached. Skipping this request.")
    return None  # Return None if max retries exceeded


def parse_questions(generated_output):
    questions = []
    current_question = None
    current_options = []
    correct_option = None
    lines = generated_output.strip().split('\n')
    
    for line in lines:
        question_match = re.match(r'\d+\)\s*(.*)', line.strip())
        option_match = re.match(r'-\s*([A-D]):\s*(.*)', line.strip())
        answer_match = re.match(r'\[Correct answer\]:\s*([A-D])', line.strip())
        
        if question_match:
            if current_question:
                questions.append({
                    "question": current_question,
                    "options": current_options,
                    "correct_option": correct_option
                })

            current_question = question_match.group(1)
            current_options = []
            correct_option = None

        elif option_match:
            option_label = option_match.group(1)
            option_text = option_match.group(2)
            current_options.append({
                "label": option_label,
                "text": option_text
            })

        elif answer_match:
            correct_option = answer_match.group(1) if answer_match.group(1) else answer_match.group(2)

    if current_question:
        questions.append({
            "question": current_question,
            "options": current_options,
            "correct_option": correct_option
        })

    return questions

results = []

for prompt_name, prompt_template in prompts:
    all_questions = []
    total_videos = len(titles)

    for idx, (title, category, transcript) in enumerate(zip(titles, categories, transcripts)):
        # Generate questions
        generated_output = generate_questions(prompt_template, category, transcript)
        parsed_questions = parse_questions(generated_output)
        
        # print the generated_output
        # print(generated_output)
        # # print the parsed_questions
        # print(parsed_questions)

        # Retry with summary if initial parse fails
        while not parsed_questions:
            print(f"Empty result, retrying for page 1, video {idx + 1}...")
            generated_output = generate_questions(prompt_template, category, transcript)
            parsed_questions = parse_questions(generated_output)

        # Append parsed questions
        all_questions.append({
            "title": title,
            "category": category,
            "multiple-choice": parsed_questions
        })
        
        # Show progress
        print(f"Progress for page {page_number} - {prompt_name} prompt: {idx + 1}/{total_videos}")

    results.extend(all_questions)
    


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for entry in results:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"All questions combined and saved to {OUTPUT_FILE}")
