click_doc_template = """I will give you the screenshot of a mobile app, the clickable UI element is labeled 
with a letter 'c' and the number <ui_element> on the screen. The tag of each element is located at the center of the 
element. Clicking on this UI element is a necessary part of proceeding with a larger task, which is to <task_description>.
In order to realize this larger task, you must first realize the current task <current_task_desc> in current screenshot.
Your task is to describe the functionality of the UI element concisely in one or two sentences. Notice that your 
description of the UI element should focus on the general function. For example, if the UI element is used to navigate 
to the chat window with John, your description should not include the name of the specific person. Just say: 
"Clicking this area will navigate the user to the chat window". Never include the tag of the
UI element in your description. You can use pronouns such as "the UI element" to refer to the element."""

input_doc_template = """I will give you the screenshot of a mobile app, which is about inputing text.
Texting in current screenshot is a necessary part of proceeding with a larger task, which is to <task_description>. 
In order to realize this larger task, you must first realize the current task <current_task_desc>. Your task is to 
describe the functionality of the UI element concisely in one or two sentences. Notice that your description of the 
UI element should focus on the general function. For example, if the change of the screenshot shows that the user 
typed "How are you?" in the chat box, you do not need to mention the actual text. Just say: "This input area is used 
for the user to type a message to send to the chat window."."""

scroll_doc_template = """I will give you the screenshot of a mobile, the scrollable UI element is labeled with a 
letter 's' and the number <ui_element> on the screen. The tag of each element is located 
at the center of  the element. Scroll this UI element is a necessary part of proceeding with a larger task, which is to 
<task_description>. In order to realize this larger task, you must first realize the current task <current_task_desc>
Your task is to describe the functionality of the UI element concisely in one or two sentences. Notice that 
your description of the UI element should be as general as possible. For example, if scrolling the UI element increases 
the contrast ratio of an image of a building, your description should be just like this: "Scrolling this area enables 
the user to tune a specific parameter of the image". Never include the tag of the UI element in your description. 
You can use pronouns such as "the UI element" to refer to the element."""

refine_doc_suffix = """\nA documentation of this UI element generated from previous demos is shown below. Your 
generated description should be based on this previous doc and optimize it. Notice that it is possible that your 
understanding of the function of the UI element derived from the given screenshots conflicts with the previous doc, 
because the function of a UI element can be flexible. In this case, your generated description should combine both.
Old documentation of this UI element: <old_doc>"""

singlepath_task_template = """You are an agent that is trained to perform some basic tasks on a smartphone. You will be given a 
smartphone screenshot. The interactive clickable UI elements on the screenshot are labeled with tags starting from "c1". 
The interactive scrollable UI elements on the screenshot are labeled with tags starting from "s1".The tag of each 
interactive element is located in the center of the element. Every screenshot I've given you is a screenshot after 
executing the correct action. 

You can call the following functions to control the smartphone:

1. click(element: str)
This function is used to click an UI element shown on the smartphone screen.
"element" is a tag assigned to an UI element shown on the smartphone screen.
A simple use case can be click(c5), which clicks the UI element labeled with "c5".

2. input(text_input: str)
This function is used to insert text input in an input field/box. text_input is the string you want to insert and must 
be wrapped with double quotation marks. A simple use case can be text("Hello, world!"), which inserts the string 
"Hello, world!" into the input area on the smartphone screen. This function is usually callable when you see a screenshot 
about text inputing.

3. scroll(element: str, direction: str)
This function is used to scroll an UI element shown on the smartphone screen, usually a scroll view or a slide bar.
"element" is a tag assigned to an UI element shown on the smartphone screen. "direction" is a string that 
represents one of the four directions: up, down, left, right. "direction" must be wrapped with double quotation 
marks.
A simple use case can be swipe(s21, "up"), which scroll up the UI element labeled with "s21".

<ui_document>
The task you need to complete is to <task_description>, to complete this task you should perform current task 
<current_task_desc>. Your past actions to proceed with this task are summarized as follows: <last_act>
Now, given the documentation and the following labeled screenshot, you need to think and call the function needed to 
proceed with the task. Your output should include three parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the next step I should do>
Action: <The function call with the correct parameters to proceed with the task.>
Summary: <Summarize your past actions along with your latest action in one or two sentences. Do not include the 
tag in your summary>
You can only take one action at a time, so please directly call the function."""

multipath_task_template = """You are an agent that is trained to perform some basic tasks on a smartphone. You will be given a 
smartphone screenshot. The interactive clickable UI elements on the screenshot are labeled with tags starting from "c1". 
The interactive scrollable UI elements on the screenshot are labeled with tags starting from "s1".The tag of each 
interactive element is located in the center of the element. Every screenshot I've given you is a screenshot after 
executing the correct action. 

You can call the following functions to control the smartphone:

1. click(element: str)
This function is used to click an UI element shown on the smartphone screen.
"element" is a tag assigned to an UI element shown on the smartphone screen.
A simple use case can be click(c5), which taps the UI element labeled with "c5".

2. input(text_input: str)
This function is used to insert text input in an input field/box. text_input is the string you want to insert and must 
be wrapped with double quotation marks. A simple use case can be text("Hello, world!"), which inserts the string 
"Hello, world!" into the input area on the smartphone screen. This function is usually callable when you see a screenshot 
about text inputing.

3. scroll(element: str, direction: str)
This function is used to scroll an UI element shown on the smartphone screen, usually a scroll view or a slide bar.
"element" is a tag assigned to an UI element shown on the smartphone screen. "direction" is a string that 
represents one of the four directions: up, down, left, right. "direction" must be wrapped with double quotation 
marks.
A simple use case can be swipe(s21, "up"), which scroll up the UI element labeled with "s21".

4. back(element:str)
This function is used to go back to the previous page. A simple use case can be back("back"), which back to the previous page.

<ui_document>
The task you need to complete is to <task_description>, to complete this task you should perform current task 
<current_task_desc>. Your past actions to proceed with this task are summarized as follows: <last_act>
Now, given the documentation and the following labeled screenshot, you need to think and call the function needed to 
proceed with the task. Your output should include three parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the next step I should do>
Action: <The function call with the correct parameters to proceed with the task. If you feel that the current page 
has no action to perform or there are no label on the screenshot or the current page is not what you want to be 
able to fulfill the task <current_task_desc>, you need to consider whether to back to the previous page.>
Summary: <Summarize your past actions along with your latest action in one or two sentences. Do not include the 
tag in your summary>
You can only take one action at a time, so please directly call the function."""

ambiguous_task_template = """You are an agent that is trained to perform some basic tasks on a smartphone. You will be given a 
smartphone screenshot. The interactive clickable UI elements on the screenshot are labeled with tags starting from "c1". 
The interactive scrollable UI elements on the screenshot are labeled with tags starting from "s1".The tag of each 
interactive element is located in the center of the element. Every screenshot I've given you is a screenshot after 
executing the correct action. 

You can call the following functions to control the smartphone:

1. click(element: str)
This function is used to click an UI element shown on the smartphone screen.
"element" is a tag assigned to an UI element shown on the smartphone screen.
A simple use case can be click(c5), which taps the UI element labeled with "c5".

2. input(text_input: str)
This function is used to insert text input in an input field/box. text_input is the string you want to insert and must 
be wrapped with double quotation marks. A simple use case can be text("Hello, world!"), which inserts the string 
"Hello, world!" into the input area on the smartphone screen. This function is usually callable when you see a screenshot 
about text inputing.

3. scroll(element: str, direction: str)
This function is used to scroll an UI element shown on the smartphone screen, usually a scroll view or a slide bar.
"element" is a tag assigned to an UI element shown on the smartphone screen. "direction" is a string that 
represents one of the four directions: up, down, left, right. "direction" must be wrapped with double quotation 
marks.
A simple use case can be swipe(s21, "up"), which scroll up the UI element labeled with "s21".

<ui_document>
The task you need to complete is to <task_description>, to complete this task you should perform current task 
<current_task_desc>. Your past actions to proceed with this task are summarized as follows: <last_act>
Now, given the documentation and the following labeled screenshot, you need to think and call the function needed to 
proceed with the task. If you don't know what to do in order to accomplish the tasks on the current page, you can think 
and act on the following questions and answers <answers>.The questions are <questions>, the answers are <answers>.
This also asks and answers questions that may have more than one similar question corresponding to a single answer.
The questions and answers are related to the pages encountered in the assignment. Questions and answers correspond by number.
Your output should include three parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the next step I should do>
Action: <The function call with the correct parameters to proceed with the task.>
Summary: <Summarize your past actions along with your latest action in one or two sentences. Do not include the 
tag in your summary>
You can only take one action at a time, so please directly call the function."""

noisy_task_template = """You are an agent that is trained to perform some basic tasks on a smartphone. You will be given a 
smartphone screenshot. The interactive clickable UI elements on the screenshot are labeled with tags starting from "c1". 
The interactive scrollable UI elements on the screenshot are labeled with tags starting from "s1".The tag of each 
interactive element is located in the center of the element. Every screenshot I've given you is a screenshot after 
executing the correct action. 

You can call the following functions to control the smartphone:

1. click(element: str)
This function is used to click an UI element shown on the smartphone screen.
"element" is a tag assigned to an UI element shown on the smartphone screen.
A simple use case can be click(c5), which clicks the UI element labeled with "c5".

2. input(text_input: str)
This function is used to insert text input in an input field/box. text_input is the string you want to insert and must 
be wrapped with double quotation marks. A simple use case can be text("Hello, world!"), which inserts the string 
"Hello, world!" into the input area on the smartphone screen. This function is usually callable when you see a screenshot 
about text inputing.

3. scroll(element: str, direction: str)
This function is used to scroll an UI element shown on the smartphone screen, usually a scroll view or a slide bar.
"element" is a tag assigned to an UI element shown on the smartphone screen. "direction" is a string that 
represents one of the four directions: up, down, left, right. "direction" must be wrapped with double quotation 
marks.
A simple use case can be swipe(s21, "up"), which scroll up the UI element labeled with "s21".

<ui_document>
The task you need to complete is to <task_description>, to complete this task you should perform current task 
<current_task_desc>. Your past actions to proceed with this task are summarized as follows: <last_act>
Now, given the documentation and the following labeled screenshot, you need to think and call the function needed to 
proceed with the task. Your output should include three parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the next step I should do>
Action: <The function call with the correct parameters to proceed with the task.>
Summary: <Summarize your past actions along with your latest action in one or two sentences. Do not include the 
tag in your summary>
You can only take one action at a time, so please directly call the function."""
