import re
import PyPDF2
import io
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import json

class ResumeAnalysisAgent:
    def __init__(self, api_key, cutoff_score=75):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.resume_text = None
        self.rag_vectorstore = None
        self.analysis_result = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weaknesses = []
        self.resume_strengths = []
        self.improvement_suggestions = {}

    def extract_text_from_pdf(self, pdf_file):
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                reader = PyPDF2.PdfReader(pdf_file_like)
            else:
                reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            return text
        
        except Exception as e:
            print(f"Error extacting text from PDf: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_file):
        try:
            if hasattr(txt_file, 'getvalue'):
                return txt_file.getvalue().decode('utf-8')
            else:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error extracting text from TXT file: {e}")
            return ""
    
    def extract_text_from_file(self, file):
        if hasattr(file, 'name'):
            file_extension = file.name.split('.')[-1].lower()
        else:
            file_extension = file.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'text':
            return self.extract_text_from_txt(file)
        else:
            print(f"Unsupported file extension: {file_extension}")
            return ""
        
    def create_rag_vector_store(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len,
            )
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    
    def create_vector_store(self, text):
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        vectorstore = FAISS.from_texts([text], embeddings)
        return vectorstore
    
    def analyze_skill(self, qa_chain, skill):
        query = f"On a scale of 0-10, how clearly does the candidate mention his proficiency in {skill}? Provide a numeric rating first, followed by reasoning"
        response = qa_chain.run(query)
        match = re.search(r"(\d{1,2})", response)
        score = int(match.group(1)) if match else 0

        reasoning  = response.split('.', 1)[1].strip() if '.' in response and len(response.split('.')) > 1 else ""

        return skill, min(score, 10), reasoning
    
    def analyze_resume_weaknesses(self):
        if not self.resume_text or not self.extracted_skills or not self.analysis_result:
            return[]
        
        weaknesses = []

        for skill in self.analysis_result.get("missing_skills", []):
            llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)
            prompt = f"""
            Analyze why the resume is weak in demonstrating proficiency in "{skill}".

            For your analysis, consider:
            1. What's missing from the resume regarding this skill?
            2. How could it be improved with specific examples?
            3. What specific action items would make this skill stand out?

            Resume Content:
            {self.resume_text[:3000]}....

            Provide your reponse in this JSON format:
            {{
                "weakness": "A concise description of what' missing or problematic(1-2 sentences)",
                "improvement_suggestions": [
                    "Specific suggestion 1",
                    "Specific suggestion 2",
                    "Specific suggestion 3"
                ],
                "example_addition": "A specific bullet point that could be added to showcase this skill"
            }}

            Return only valid JSON, no other text.
            """

            response = llm.invoke(prompt)
            weakness_content = response.content.strip()

            try:
                weakness_data = json.loads(weakness_content)
                
                weakness_detail = {
                    "skill": skill,
                    "score": self.analysis_result.get("skill_scores", {}).get(skill, 0),
                    "detail": weakness_data.get("weakness", "No specific details provided"),
                    "suggestions": weakness_data.get("example_addition", "")
                }

                weaknesses.append(weakness_detail)

                self.improvement_suggestions[skill] = {
                    "suggestions": weakness_data.get("imrpovement_suggestions", []),
                    "example": weakness_data.get("example_addition", "")
                }
            except json.JSONDecoder:

                weaknesses.append({
                    "skill": skill,
                    "score": self.analysis_result.get("skill_scores", {}).get(skill, 0),
                    "detail": weakness_content[:200]
                })
        
        self.resume_weaknesses = weaknesses
        return weaknesses
    
    def extract_skills_from_jd(self, jd_text):
        try:
            llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)
            prompt = f"""
            Extract a comprehensive list of technical skills, technologies, and competencies required from this job description.
            Format the output as a Python list of things. Only include the list, nothing else.

            Job Description:
            {jd_text}
            """

            response = llm.invoke(prompt)
            skills_text = response.content

            match = re.search(r'\[(.*?)\]', skills_text, re.DOTALL)
            if match:
                skills_text = match.group(0)
            

            try:
                skills_list = eval(skills_text)
                if isinstance(skills_list, list):
                    return skills_list
            
            except:
                pass

            skills = []
            for line in skills_text.split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    skill = line[2:].strip()
                    if skill:
                        skills.append(skill)
                elif line.startswith('"') and line.endswith('"'):
                    skill = line.strip('"')
                    if skill:
                        skills.append(skill)
            
            return skills
        except Exception as e:
            print(f"Error extracting skills from the job description: {e}")
            return []
    
    def semantic_skill_analysis(self, resume_text, skills):
        vectorstore = self.create_rag_vector_store(resume_text)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm = ChatOpenAI(model="gpt-4o", api_key = self.api_key),
            retriever = retriever,
            return_source_document = False
        )

        skill_scores = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda skill: self.analyze_skill(qa_chain, skill), skills))
        
        for skill, score, reasoning, in results:
            skill_scores[skill] = score
            skill_reasoning[skill] = reasoning
            total_score += score
            if score <= 5:
                missing_skills.append(skill)

        overall_score = int((total_score / (10 * len(skills))) * 100)
        selected = overall_score >= self.cutoff_score

        reasoning = "Candidate evaluated based on explicict resume content using semantic similarity and clear numeric scoring."
        strengths = [skill for skill, score in skill_scores.items() if score >= 7]
        imrpovement_areas = missing_skills if not selected else []

        self.resume_strengths = strengths

        return {
            "overall_score": overall_score,
            "skill_scores": skill_scores,
            "skill_reasoning": skill_reasoning,
            "selected": selected,
            "reasoning": reasoning,
            "missing_skills": missing_skills,
            "strengths": strengths,
            "improvement_areas": imrpovement_areas
        }
    
    def analyze_resume(self, resume_file, role_requirements = None, custom_jd = None):
        self.resume_text = self.extract_text_from_file(resume_file)


        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
            tmp.write(self.resume_text)
            self.resume_file_path = tmp.name
        
        self.rag_vectorstore = self.create_rag_vector_store(self.resume_text)

        if custom_jd:
            self.jd_text = self.extract_text_from_file(custom_jd)
            self.extracted_skills = self.extract_skills_from_jd(self.jd_text)

            self.analysis_result = self.semantic_skill_analysis(self.resume_text, self.extracted_skills)
        
        elif role_requirements:
            self.extracted_skills = role_requirements

            self.analysis_result = self.semantic_skill_analysis(self.resume_text, role_requirements)

        if self.analysis_result and "missing_skills" in self.analysis_result and self.analysis_result["missing_skills"]:
            self.analyze_resume_weaknesses()

            self.analysis_result["detailed_weaknesses"] = self.resume_weaknesses
        
        return self.analysis_result
    
    def ask_question(self, question):
        if not self.rag_vectorstore or not self.resume_text:
            return "Please analyze a resume first."
        
        retriever = self.rag_vectorstore.as_retriever(
            search_kwargs = {"k": 3}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm = ChatOpenAI(model = "gpt-4o", api_key=self.api_key),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )

        response = qa_chain.run(question)
        return response
    
    def generate_interview_questions(self, question_types, difficulty, num_questions):
        if not self.resume_text or not self.extracted_skills:
            return[]
        
        try:
            llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)

            context = f"""
            Resume Content:
            {self.resume_text[:2000]}...

            Skills to focus on: {', '.join(self.extracted_skills)}

            Strengths: {', '.join(self.analysis_result.get('strengths', []))}

            Areas for improvement: {', '.join(self.analysis_result.get('missing_skills', []))}
            """

            prompt = f"""
            Generate {num_questions} personalized {difficulty.lower()} level interview questions for this candidate 
            based on their resume and skills. Include only the following question types: {", ".join(question_types)}
            
            For each question:
            1. Clearly label the question type
            2. Make the question specific to their background and skills
            3. For coding question, include a clear problem statement

            {context}

            Format the response as a list of tuples with the question type and the question itself.
            Each tuple should be in the format: ("Question Type", "Full Question Text")
            """
            response = llm.invoke(prompt)
            questions_text = response.content

            questions = []
            pattern = r'[("]([^"]+)[",)\s]+[(",\s]+([^"]+)[")\s]+'
            matches = re.findall(pattern, questions_text, re.DOTALL)

            for match in matches:
                if len(matches) >= 2:
                    question_type = match[0].strip()
                    question = match[1].strip()

                    for requested_type in question_types:
                        if requested_type.lower() in question_type.lower():
                            questions.append((requested_type, question))
                            break

            if not questions:
                lines = questions_text.split('\n')
                current_type = None
                current_question = ""

                for line in lines:
                    line = line.strip()
                    if any(t.lower() in line.lower() for t in question_types) and not current_question:
                        current_type = next((t for t in question_types if t.lower() in line.lower()), None)
                        if ":" in line:
                            current_question = line.split(":", 1)[1].strip()
                    elif current_type and line:
                        current_question += " " + line
                    elif current_type and current_question:
                        questions.append((current_type, current_question))
                        current_type = None
                        current_question = ""
            
            questions = questions[:num_questions]

            return questions
        
        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return []
    
    def improve_resume(self, improvement_areas, target_role=""):
        if not self.resume_text:
            return {}
        
        try:
            improvements = {}

            for area in improvement_areas:
                if area == "Skills Highlighting" and self.resume_weaknesses:
                    skill_improvements = {
                        "description": "Your resume needs to better highlight key skills that are important for the role.", "specific": []
                    }

                    before_after_examples = {}
                    
                    for weakness in self.resume_weaknesses:
                        skill_name = weakness.get("skill", "")
                        if "suggestions" in weakness and weakness["suggestions"]:
                            for suggestion in weakness["suggestions"]:
                                skill_improvements["specific"].append(f"**{skill_name}**: {suggestion}")
                            
                        if "example" in weakness and weakness["example"]:
                            resume_chunks = self.resume_text.split('\n\n')
                            relevant_chunk = ""

                            for chunk in resume_chunks:
                                if skill_name.lower() in chunk.lower() or "experience" in chunk.lower():
                                    relevant_chunk = chunk
                                    break
                            
                            if relevant_chunk:
                                before_after_examples = {
                                    "before": relevant_chunk.strip(),
                                    "after": relevant_chunk.strip()+"\n." + weakness["example"]
                                }

                    if before_after_examples:
                        skill_improvements["before_after"] = before_after_examples

                    improvements["Skill Highlighting"] = skill_improvements

            remaining_areas = [area for area in improvement_areas if area not in improvements]

            if remaining_areas:
                llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)

                weakness_text = ""
                if self.resume_weaknesses:
                    weakness_text = "Resume Weaknesses:\n"
                    for i, weakness in enumerate(self.resume_weaknesses):
                        weakness_text += f"{i+1}. {weakness['skill']}: {weakness['detail']}\n"

                        if "suggestions" in weakness:
                            for j, sugg in enumerate(weakness["suggestions"]):
                                weakness_text += f"   - {sugg}\n"

                context = f"""
                Resume Content:
                {self.resume_text}

                Skills to focus on: {', '.join(self.extracted_skills)}

                Strengths: {', '.join(self.analysis_result.get('strengths', []))}

                Areas of Improvement: {', '.join(self.analysis_result.get('missing_skills', []))}

                {weakness_text}

                Target role: {target_role if target_role else "Not specified"}
                """

                prompt = f"""
                Provide detailed suggestions to improve this resume in the following areas: {', '.join(remaining_areas)}.

                {context}

                For each improvement area, provide:
                1. A general description of what needs improvement
                2. 3-5 specific actionable suggestions
                3. Where relevant, provide a before/after example

                Format the response as a JSON object with improvement areas as keys, each containing:
                - "description": general description
                - "specific": list of specific suggestions
                - "before_after": (where applicable) a dict with "before" and "after" examples

                Only include the requested improvement area that aren't already covered.
                Focus particularly on addressing the resume weaknesses identified.
                """

                response = llm.invoke(prompt)

                ai_improvements = {}

                json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', response.content)

                if json_match:
                    try:
                        ai_improvements = json.loads(json_match.group(1))

                        improvements.update(ai_improvements)
                    except json.JSONDecodeError:
                        pass
                
                if not ai_improvements:
                    sections = response.content.split("##")

                    for section in sections:
                        if not section.strip():
                            continue

                        lines = section.strip().split("\n")
                        area = None

                        for line in lines:
                            if not area and line.strip():
                                area = line.strip()
                                improvements[area] = {
                                    "description": "",
                                    "specific": []
                                }
                            
                            elif area and "specific" in improvements[area]:
                                if line.strip().startswith("- "):
                                    improvements[area]["specific"].append(line.strip()[2:])
                                elif not improvements[area]["description"]:
                                    improvements[area]["description"] += line.strip()

            for area in improvement_areas:
                if area not in improvements:
                    improvements[area] = {
                        "description": f"Improvements needed in {area}",
                        "specific": ["Review and enhance this section"]
                    }

            return improvements
        
        except Exception as e:
            print(f"Error generating resume improvements: {e}")
            return {area: {"description": "Error generating suggestions", "specific": []} for area in improvement_areas}

