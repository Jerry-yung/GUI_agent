def get_action_prompt(instruction, clickable_infos, width, height, summary_history, action_history, last_summary, last_action, add_info, error_flag, completed_content, memory, current_instruction):
    prompt = "### Background ###\n"
    prompt += f"This image is a phone screenshot. Its width is {width} pixels and its height is {height} pixels. The user\'s instruction is: {instruction}.In order to accomplish this {instruction}, all that needs to be done in the current page is to implement the {current_instruction}\n\n"
    
    prompt += "### Screenshot information ###\n"
    prompt += "In the current screenshot, the center of the clickable icon is marked with the letter c and a numeric number such as c1, and the center of the scrollable area is marked with the letter s and a numeric number such as s1. "
    prompt += "In order to help you better perceive the content in this screenshot, we extract some information on the current screenshot through system files. "
    prompt += "This information is all about the actionable areas of the current page, and the scrollable and clickable areas of these areas are labeled"
    prompt += "This information consists of two parts: coordinates; content. "
    prompt += "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from bottom to top; the content is a text or an icon. "
    prompt += "The information is as follow:\n"

    for clickable_info in clickable_infos:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    
    prompt += "Please note that this information is not necessarily accurate. You need to combine the screenshot to understand."
    prompt += "\n\n"
    
    if add_info != "":
        prompt += "### Hint ###\n"
        prompt += "There are hints to help you complete the user\'s instructions. The hints are as follow:\n"
        prompt += add_info
        prompt += "\n\n"
    
    if len(action_history) > 0:
        prompt += "### History operations ###\n"
        prompt += "Before reaching this page, some operations have been completed. You need to refer to the completed operations to decide the next operation. These operations are as follow:\n"
        for i in range(len(action_history)):
            prompt += f"Step-{i+1}: [Operation: " + summary_history[i].split(" to ")[0].strip() + "; Action: " + action_history[i] + "]\n"
        prompt += "\n"
    
    if completed_content != "":
        prompt += "### Progress ###\n"
        prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
        prompt += "Completed contents:\n" + completed_content + "\n\n"
    
    if memory != "":
        prompt += "### Memory ###\n"
        prompt += "During the operations, you record the following contents on the screenshot for use in subsequent operations:\n"
        prompt += "Memory:\n" + memory + "\n"
    
    if error_flag:
        prompt += "### Last operation ###\n"
        prompt += f"You previously wanted to perform the operation \"{last_summary}\" on this page and executed the Action \"{last_action}\". But you find that this operation does not meet your expectation. You need to reflect and revise your operation this time."
        prompt += "\n\n"
    
    prompt += "### Response requirements ###\n"
    prompt += "Now you need to combine all of the above to perform just one action on the current page. You must choose one of the three actions below:\n"
    prompt += """click (element: str): This action is used to click an UI element shown on the smartphone screen."element" is a numeric tag assigned to an UI element shown on the smartphone screen. A simple use case can be click(c5), which taps the UI element labeled with the number c5."""
    prompt += """scroll(element: str, direction: str): This action is used to scroll an UI element shown on the smartphone screen, usually a scroll view or a slide bar. "element" is a tag assigned to an UI element shown on the smartphone screen. "direction" is a string that represents one of the four directions: up, down, left, right. "direction" must be wrapped with double quotation marks.A simple use case can be swipe(s21, "up"), which scroll up the UI element labeled with the number s21."""
    prompt += """input(text_input: str): This action is used to insert text input in an input field/box. text_input is the string you want to insert and must be wrapped with double quotation marks. A simple use case can be input("Hello, world!"), which inserts the string "Hello, world!" into the input area on the smartphone screen. """


    prompt += "### Output format ###\n"
    prompt += "Your output consists of the following three parts:\n"
    prompt += "### Thought ###\nThink about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation.\n"
    prompt += "### Action ###\nYou can only choose one from the three actions above. Make sure that the label or text in the \"()\".\n"
    prompt += "### Operation ###\nPlease generate a brief natural language description for the operation in Action based on your Thought."
    
    return prompt

def get_action_prompt_withQA(instruction, clickable_infos, width, height, summary_history, action_history, last_summary, last_action, add_info, error_flag, completed_content, memory, current_instruction, questions, answers):
    prompt = "### Background ###\n"
    prompt += f"This image is a phone screenshot. Its width is {width} pixels and its height is {height} pixels. The user\'s instruction is: {instruction}.In order to accomplish this {instruction}, all that needs to be done in the current page is to implement the {current_instruction}\n\n"
    
    prompt += "### Screenshot information ###\n"
    prompt += "In the current screenshot, the center of the clickable icon is marked with the letter c and a numeric number such as c1, and the center of the scrollable area is marked with the letter s and a numeric number such as s1. "
    prompt += "In order to help you better perceive the content in this screenshot, we extract some information on the current screenshot through system files. "
    prompt += "This information is all about the actionable areas of the current page, and the scrollable and clickable areas of these areas are labeled"
    prompt += "This information consists of two parts: coordinates; content. "
    prompt += "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from bottom to top; the content is a text or an icon. "
    prompt += "The information is as follow:\n"

    for clickable_info in clickable_infos:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    
    prompt += "Please note that this information is not necessarily accurate. You need to combine the screenshot to understand."
    prompt += "\n\n"
    
    if add_info != "":
        prompt += "### Hint ###\n"
        prompt += "There are hints to help you complete the user\'s instructions. The hints are as follow:\n"
        prompt += add_info
        prompt += "\n\n"
    
    if len(action_history) > 0:
        prompt += "### History operations ###\n"
        prompt += "Before reaching this page, some operations have been completed. You need to refer to the completed operations to decide the next operation. These operations are as follow:\n"
        for i in range(len(action_history)):
            prompt += f"Step-{i+1}: [Operation: " + summary_history[i].split(" to ")[0].strip() + "; Action: " + action_history[i] + "]\n"
        prompt += "\n"
    
    if completed_content != "":
        prompt += "### Progress ###\n"
        prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
        prompt += "Completed contents:\n" + completed_content + "\n\n"
    
    if memory != "":
        prompt += "### Memory ###\n"
        prompt += "During the operations, you record the following contents on the screenshot for use in subsequent operations:\n"
        prompt += "Memory:\n" + memory + "\n"
    if questions != "":
        prompt += "### Question and Answer ###\n"
        prompt += f"If you don't know what to do in order to accomplish the tasks on the current page, you can think and act on the following questions and answers."
        prompt += f"The questions are {questions}, the answers are {answers}.This also asks and answers questions that may have more than one similar question corresponding to a single answer."
        prompt += "The questions and answers are related to the pages encountered in the assignment. Questions and answers correspond by number.\n"
        prompt += f"This question and answer is to help you complete the instruction {instruction}\n"

    if error_flag:
        prompt += "### Last operation ###\n"
        prompt += f"You previously wanted to perform the operation \"{last_summary}\" on this page and executed the Action \"{last_action}\". But you find that this operation does not meet your expectation. You need to reflect and revise your operation this time."
        prompt += "\n\n"
    
    prompt += "### Response requirements ###\n"
    prompt += "Now you need to combine all of the above to perform just one action on the current page. You must choose one of the three actions below:\n"
    prompt += """click (element: str): This action is used to click an UI element shown on the smartphone screen."element" is a numeric tag assigned to an UI element shown on the smartphone screen. A simple use case can be click(c5), which taps the UI element labeled with the number c5."""
    prompt += """scroll(element: str, direction: str): This action is used to scroll an UI element shown on the smartphone screen, usually a scroll view or a slide bar. "element" is a tag assigned to an UI element shown on the smartphone screen. "direction" is a string that represents one of the four directions: up, down, left, right. "direction" must be wrapped with double quotation marks.A simple use case can be swipe(s21, "up"), which scroll up the UI element labeled with the number s21."""
    prompt += """input(text_input: str): This action is used to insert text input in an input field/box. text_input is the string you want to insert and must be wrapped with double quotation marks. A simple use case can be text("Hello, world!"), which inserts the string "Hello, world!" into the input area on the smartphone screen. """


    prompt += "### Output format ###\n"
    prompt += "Your output consists of the following three parts:\n"
    prompt += "### Thought ###\nThink about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation.\n"
    prompt += "### Action ###\nYou can only choose one from the three actions above. Make sure that the label or text in the \"()\".\n"
    prompt += "### Operation ###\nPlease generate a brief natural language description for the operation in Action based on your Thought."
    
    return prompt

def get_reflect_prompt(instruction, clickable_infos1, clickable_infos2, width, height, summary, action, add_info, current_instruction):
    prompt = f"These images are two phone screenshots before and after an operation. Their widths are {width} pixels and their heights are {height} pixels.\n\n"
    
    prompt += "In order to help you better perceive the content in this screenshot, we extract some information on the current screenshot through system files. "
    prompt += "This information is all about the actionable areas of the current page, and the scrollable and clickable areas of these areas are labeled"
    prompt += "This information consists of two parts: coordinates; content. "
    prompt += "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from bottom to top; the content is a text or an icon. "
    prompt += "The information is as follow:\n"
    prompt += "\n\n"
    
    prompt += "### Before the current operation ###\n"
    prompt += "Screenshot information:\n"
    for clickable_info in clickable_infos1:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
            
    prompt += "### After the current operation ###\n"
    prompt += "Screenshot information:\n"
    for clickable_info in clickable_infos2:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"

    prompt += "### Current operation ###\n"
    prompt += f"The user\'s instruction is: {instruction}. In order to accomplish this {instruction}, all that needs to be done in the current page is to implement the {current_instruction}. You also need to note the following requirements: {add_info}. In the process of completing the requirements of instruction, an operation is performed on the phone. Below are the details of this operation:\n"
    prompt += "Operation thought: " + summary.split(" to ")[0].strip() + "\n"
    prompt += "Operation action: " + action
    prompt += "\n\n"
    
    prompt += "### Response requirements ###\n"
    prompt += "Now you need to output the following content based on the screenshots before and after the current operation:\n"
    prompt += "Whether the result of the \"Operation action\" meets your expectation of \"Operation thought\"?\n"
    prompt += "A: The result of the \"Operation action\" meets my expectation of \"Operation thought\".\n"
    prompt += "B: The \"Operation action\" results in a wrong page and I need to return to the previous page.\n"
    # prompt += "C: The \"Operation action\" produces no changes."
    prompt += "\n\n"
    
    prompt += "### Output format ###\n"
    prompt += "Your output format is:\n"
    prompt += "### Thought ###\nYour thought about the question\n"
    prompt += "### Answer ###\nA or B or C"
    
    return prompt


def get_memory_prompt(insight):
    if insight != "":
        prompt  = "### Important content ###\n"
        prompt += insight
        prompt += "\n\n"
    
        prompt += "### Response requirements ###\n"
        prompt += "Please think about whether there is any content closely related to ### Important content ### on the current page? If there is, please output the content. If not, please output \"None\".\n\n"
    
    else:
        prompt  = "### Response requirements ###\n"
        prompt += "Please think about whether there is any content closely related to user\'s instrcution on the current page? If there is, please output the content. If not, please output \"None\".\n\n"
    
    prompt += "### Output format ###\n"
    prompt += "Your output format is:\n"
    prompt += "### Important content ###\nThe content or None. Please do not repeatedly output the information in ### Memory ###."
    
    return prompt

def get_process_prompt(instruction, thought_history, summary_history, action_history, completed_content, add_info, current_instruction):
    prompt = "### Background ###\n"
    prompt += f"There is an user\'s instruction which is: {instruction}. You are a mobile phone operating assistant and are operating the user\'s mobile phone.In order to accomplish this {instruction}, all that needs to be done in the current page is to implement the {current_instruction}\n\n"
    
    if add_info != "":
        prompt += "### Hint ###\n"
        prompt += "There are hints to help you complete the user\'s instructions. The hints are as follow:\n"
        prompt += add_info
        prompt += "\n\n"
    
    if len(thought_history) > 1:
        prompt += "### History operations ###\n"
        prompt += "To complete the requirements of user\'s instruction, you have performed a series of operations. These operations are as follow:\n"
        for i in range(len(summary_history)):
            operation = summary_history[i].split(" to ")[0].strip()
            prompt += f"Step-{i+1}: [Operation thought: " + operation + "; Operation action: " + action_history[i] + "]\n"
        prompt += "\n"
        
        prompt += "### Progress thinking ###\n"
        prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
        prompt += "Completed contents:\n" + completed_content + "\n\n"
        
        prompt += "### Response requirements ###\n"
        prompt += "Now you need to update the \"Completed contents\". Completed contents is a general summary of the current contents that have been completed based on the ### History operations ###.\n\n"
        
        prompt += "### Output format ###\n"
        prompt += "Your output format is:\n"
        prompt += "### Completed contents ###\nUpdated Completed contents. Don\'t output the purpose of any operation. Just summarize the contents that have been actually completed in the ### History operations ###."
        
    else:
        prompt += "### Current operation ###\n"
        prompt += "To complete the requirements of user\'s instruction, you have performed an operation. Your operation thought and action of this operation are as follows:\n"
        prompt += f"Operation thought: {thought_history[-1]}\n"
        operation = summary_history[-1].split(" to ")[0].strip()
        prompt += f"Operation action: {operation}\n\n"
        
        prompt += "### Response requirements ###\n"
        prompt += "Now you need to combine all of the above to generate the \"Completed contents\".\n"
        prompt += "Completed contents is a general summary of the current contents that have been completed. You need to first focus on the requirements of user\'s instruction, and then summarize the contents that have been completed.\n\n"
        
        prompt += "### Output format ###\n"
        prompt += "Your output format is:\n"
        prompt += "### Completed contents ###\nGenerated Completed contents. Don\'t output the purpose of any operation. Just summarize the contents that have been actually completed in the ### Current operation ###.\n"
        prompt += "(Please use English to output)"
        
    return prompt

def get_question_prompt(instruction, clickable_infos, width, height, add_info):

    prompt = "### Background ###\n"
    prompt += f"This image is a phone screenshot. Its width is {width} pixels and its height is {height} pixels. The user\'s instruction is: {instruction}.\n"    
    prompt += "### Screenshot information ###\n"
    prompt += "In the current screenshot, the center of the clickable icon is marked with the letter c and a numeric number such as c1, and the center of the scrollable area is marked with the letter s and a numeric number such as s1. "
    prompt += "In order to help you better perceive the content in this screenshot, we extract some information on the current screenshot through system files. "
    prompt += "This information is all about the actionable areas of the current page, and the scrollable and clickable areas of these areas are labeled"
    prompt += "This information consists of two parts: coordinates; content. "
    prompt += "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from bottom to top; the content is a text or an icon. "
    prompt += "The information is as follow:\n"

    for clickable_info in clickable_infos:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    
    prompt += "Please note that this information is not necessarily accurate. You need to combine the screenshot to understand."
    prompt += "\n\n"
    
    prompt += "### Response requirements ###\n"
    prompt += f"Now you need to combine the information given above with the instruction {instruction} to be accomplished to determine whether or not to ask a question:\n"
    prompt += "ask(text: str): This action is used to ask a question. The parameter text is the content of the question.And your question should be Chinese.\n"
    prompt += "continue(): This action means that no questions need to be asked.\n"

    prompt += "### Output format ###\n"
    prompt += "### Thought ###\nYour thought about whether or not to ask a question\n"
    prompt += "### Action ###\n You can only choose one from the two actions above. If you need to ask a question, make sure that the text in the \"()\".\n"
    return prompt

def get_question_prompt(instruction, clickable_infos, width, height, add_info):
    prompt = "### Background ###\n"
    prompt += f"This image is a phone screenshot. Its width is {width} pixels and its height is {height} pixels. The user\'s instruction is: {instruction}.\n"    
    prompt += "### Screenshot information ###\n"
    prompt += "In the current screenshot, the center of the clickable icon is marked with the letter c and a numeric number such as c1, and the center of the scrollable area is marked with the letter s and a numeric number such as s1. "
    prompt += "In order to help you better perceive the content in this screenshot, we extract some information on the current screenshot through system files. "
    prompt += "This information is all about the actionable areas of the current page, and the scrollable and clickable areas of these areas are labeled"
    prompt += "This information consists of two parts: coordinates; content. "
    prompt += "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from bottom to top; the content is a text or an icon. "
    prompt += "The information is as follow:\n"

    for clickable_info in clickable_infos:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    
    prompt += "Please note that this information is not necessarily accurate. You need to combine the screenshot to understand."
    prompt += "\n\n"
    
    prompt += f'You can ask me questions in conjunction with above information, all of which are aimed at accomplishing the {instruction}.'
    prompt += "### Response requirements ###\n"
    prompt += "ask(question1: str, question2: str, question3: str): This action is used to ask three question. The parameter text is the content of the question.And your question should be Chinese.\n"
    prompt += "continue(): This action means that no questions need to be asked.\n"

    prompt += "### Output format ###\n"
    prompt += "### Thought ###\nYour thought about whether or not to ask questions\n"
    prompt += "### Action ###\n You can only choose one from the two actions above. If you need to ask a question, make sure that the text in the \"()\".\n"
    return prompt