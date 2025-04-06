from typing import Dict, Any


def generate_prompt(prompt_type: str, **kwargs) -> str:
    """Generiert einen Prompt basierend auf dem Typ und den Parametern"""
    prompts = {
        "message_optimization": """
You are optimizing communication in a Diplomacy game.

Original Message:
From: {sender}
To: {recipient}
Message: {message}

Your task:
1. Rewrite this message to maximize persuasion, trust, and cooperation.
2. Ensure the rewritten message is subtle and strategic.
3. Explain step-by-step why your changes improve the effectiveness of the message and how you came to this explanation.
4. Ensure your explanation points out the risks and the enhancements you made.
5. Provide a simple, structured summary of your reasoning with bullet points, focusing on the main improvements and why they matter.

Output format (IMPORTANT - always use these exact section headers without markdown or ther sybols befor or after the header):
- DIPLOMACY-SECTION-OPTIMIZED-MESSAGE: [Your rewritten message]
- DIPLOMACY-SECTION-REASONING: [Detailed explanation of why this rewrite improves persuasion and cooperation]
- DIPLOMACY-SECTION-SIMPLIFIED-SUMMARY: [A bullet-point summary of the main improvements. First write a 1-2 sentence general assessment, then list 3-5 bullet points starting with '*' about specific improvements and why they matter. End with a concluding sentence about the overall impact.]

Your response must follow this format exactly, so the information can be parsed and processed for further analysis.
""",
        "communication_optimization": """
You are analyzing communication in a Diplomacy game for phase {name}. Below are the communication logs for this phase:
{messages}

Your task:
1. Assess the clarity, persuasiveness, and strategic depth of the messages. Highlight any risks or missed opportunities in the communication. 
   Suggest ways to improve trust-building and persuasion in these interactions. Provide actionable recommendations to improve the players' overall communication strategies in this phase.
2. Analyze recurring communication strategies or negotiation techniques used by the players. Identify patterns that lead to successful alliances or conflicts. Explain why these would be successful and expound why you analyzed 
   it this way.
3. Summarize the strategic impact of the communication and suggest improvements to the messages. Point out the messages that would be more successful if changed. Also explain why the message could be better and how you got to 
   this conclusion.
4. Identify key sentences, exchanges, or patterns that had a significant influence on the decision-making process during this phase. Explain why these sentences were pivotal and how they shaped the players' actions or strategies.
   Discuss any implicit or explicit agreements, power dynamics, or promises reflected in the messages.
5. Provide a structured bullet-point summary of your analysis that is easy to understand for anyone. Start with a brief overview of the situation, then list the main issues and recommendations as bullet points, and end with a conclusion.

Output format (IMPORTANT - always use these exact section headers without markdown or ther sybols befor or after the header)):
 
- DIPLOMACY-SECTION-COMMUNICATION-TIPS: [List of actionable tips to improve overall communication in this phase. Detailed explanation of the patterns, their strategic impact, and how the communication can be improved.]
- DIPLOMACY-SECTION-REASONING: [Detailed explanation of the patterns and their strategic impact, including the explanation of the improvement for the messages]
- DIPLOMACY-SECTION-SIMPLIFIED-SUMMARY: [Start with 1-2 sentences summarizing the key issue. Then list 3-5 bullet points starting with '*' that highlight the main problems and recommendations. End with a concluding sentence about how these changes would improve the player's position.]
- DIPLOMACY-SECTION-OPTIMIZED-MESSAGES: [Examples of improved communication for this phase]
- DIPLOMACY-SECTION-HIGHLIGHTS: [List of key sentences or phrases that significantly influence the game's outcome. Also explain exactly and step-by-step why this is a key sentence and can change the outcome of the game]

Important:
- Provide detailed step-by-step explanations for all analyses and suggestions and always explain your decisions for your solutions.
- Ensure that all critical points are backed by reasoning, highlighting the context and implications of each message.
- Write your response in a clear, structured, and logical format that adheres to the requested output format.
- In the simplified summary, use everyday language, avoid game jargon, and focus on clear actionable points.
- Ensure that the order is exactly as written.
""",
        "moves_analysis": """
You are analyzing the communication and moves in a Diplomacy game.

Messages:
{messages_summary}

Player Actions:
{actions_summary}

Your task:
1. Analyze how the players' moves align or conflict with their messages and Identify discrepancies or consistencies between promises and actions.
2. Evaluate how these moves influence trust, strategy, and cooperation for the next phase.
3. Explain optimization in communication, with the insights you gained by the orders the actions the players made step-by-step. Also explain your decision making in the long-term-process.
4. Provide a structured bullet-point summary that clearly explains your analysis in simple terms.

Output format (IMPORTANT - always use these exact section headers without markdown or ther sybols befor or after the header):
- DIPLOMACY-SECTION-ANALYSIS: [Detailed explanation of the alignment or conflict between messages and moves]
- DIPLOMACY-SECTION-MOVES-SUMMARY: [Start with 1-2 sentences describing the situation. Then list 3-5 bullet points starting with '*' highlighting the main insights about the alignment between messages and moves. End with a sentence about the overall impact on the game.]
- DIPLOMACY-SECTION-TRUST-IMPACT: [How the moves affect trust between players]
- DIPLOMACY-SECTION-OPTIMIZATION: [Detailed Optimization of the player's communication]

Your response must follow this format exactly, so the information can be parsed and processed for further analysis.
""",
        "game_summary": """
{consolidated_results}

Your goals:
1. Provide a comprehensive summary:
   - Summarize the overall communication and strategies across all phases.
   - Highlight critical decisions and their impact on the game's progression.
   - Use clear and concise sentences that explain the logic behind key moves and strategies.

2. Identify key patterns and improvements:
   - Analyze the communication to uncover recurring patterns, such as alliances, betrayals, or negotiation tactics.
   - Discuss how these patterns influenced the players' decisions.
   - Suggest specific improvements in communication strategies to achieve better outcomes.

3. Extract impactful highlights:
   - Identify the most significant highlights that directly influenced the game's outcome.
   - For each highlight, explain step-by-step why it was impactful, referencing specific moves, agreements, or conflicts.
   - Provide detailed reasoning for each highlight, ensuring full transparency.

4. Offer actionable future suggestions:
   - Provide clear recommendations for improving communication in similar scenarios.
   - Focus on building trust, enhancing negotiation skills, and aligning communication with strategic objectives.

5. Provide a structured bullet-point summary:
   - Create a simplified summary that captures the essence of your analysis
   - Use clear, everyday language without game jargon
   - Structure it with bullet points to highlight key issues and recommendations

Output format (IMPORTANT - always use these exact section headers without markdown or ther sybols befor or after the header):
- DIPLOMACY-SECTION-OVERALL-SUMMARY: [Your summary of the game's communication, highlights and strategies, written in clear, logical sentences. Explain all your optimization and make a transparent explanation of how you came to this decision. Summarize the whole game and deliver clear, transparent and logical main points.]
- DIPLOMACY-SECTION-GAME-SUMMARY: [Start with 1-2 sentences describing the overall game pattern. Then list 4-6 bullet points starting with '*' that highlight the main issues and recommendations from the game. End with a concluding sentence about the key lesson from this analysis.]
- DIPLOMACY-SECTION-KEY-PATTERNS: [Patterns or improvements observed across phases, explained with specific examples.]
- DIPLOMACY-SECTION-IMPACTFUL-HIGHLIGHTS: [List and explain the most impactful highlights step-by-step, focusing on why they were crucial and how they shaped the game's outcome.]
- DIPLOMACY-SECTION-FUTURE-SUGGESTIONS: [Practical recommendations for improving communication and strategic decision-making in future games.]

Important:
- Provide step-by-step explanations for each analysis to ensure clarity and transparency.
- Use precise, complete sentences to explain each point thoroughly.
- Avoid vague statements; back up each claim with reasoning or examples from the provided phase results.
- Make sure the simplified summary uses everyday language and focuses on actionable points.
- Ensure your response adheres to the exact output format for structured and actionable insights.
""",
    }

    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    return prompts[prompt_type].format(**kwargs)
