import json
import datetime
import pymupdf4llm
import time
from pathlib import Path
import os
import traceback
from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from vllm import LLM, SamplingParams

# Set HuggingFace token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "Hugging_Face_Token"

# Global variable for model selection
SELECTED_MODEL = "llama"  # Change to "llama" or "phi"
 
class ModelManager:
    _instance = None
    _model = None
    _sampling_params = None 
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize_model(cls):
        if cls._model is None:
            print(f"Initializing {SELECTED_MODEL} model...")
            
            if SELECTED_MODEL == "mistral":
                cls._model = LLM(
                    model="mistralai/Ministral-8B-Instruct-2410",
                    trust_remote_code=True,
                    tensor_parallel_size=1
                )
                cls._sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=8192,
                )
                
            elif SELECTED_MODEL == "llama":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                
                model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                cls._model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                )
                
            elif SELECTED_MODEL == "phi":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                
                model_name = "microsoft/Phi-3.5-MoE-instruct"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                cls._model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                )
            
            print(f"{SELECTED_MODEL} model initialized successfully")
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls.initialize_model()
        return cls._model, cls._sampling_params


class SinglePassPaperAnalyzer:
    def __init__(self):
        self.paper_text = None
        self.execution_times = {
            "single_pass_analysis": 0,
            "total_time": 0
        }
        self.model_choice = SELECTED_MODEL
        # Get the already initialized model
        self.model, self.sampling_params = ModelManager.get_model()

    def extract_text_from_pdf(self, filename: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        try:
            self.paper_text = pymupdf4llm.to_markdown(filename)
            return self.paper_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def _get_model_response(self, prompt: str) -> str:
        """Unified method to get response from selected model"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in analyzing research papers."},
                {"role": "user", "content": prompt}
            ]
            
            if self.model_choice == "mistral":
                outputs = self.model.chat(messages, sampling_params=self.sampling_params)
                response = outputs[0].outputs[0].text
                
            else:  # llama or phi
                generation_args = {
                    "max_new_tokens": 8000,
                    "return_full_text": False,
                    "temperature": 0.0,
                }
                output = self.model(messages, **generation_args)
                response = output[0]['generated_text']
            
            return response
            
        except Exception as e:
            print(f"Error generating response with {self.model_choice}: {str(e)}")
            return ""

    def analyze_paper(self, filename):
        """Perform comprehensive single-pass analysis of the paper"""
        if not self.paper_text:
            text = self.extract_text_from_pdf(filename)
        else:
            text = self.paper_text
            
        if not text:
            raise Exception("Failed to extract text from PDF")
        start_time = time.time()

        comprehensive_prompt = f"""
        Analyze this research paper and provide a comprehensive evaluation.
        Paper text: {text}

        Follow these guidelines:

        1. Identify ALL claims in the paper where each claim:
           - Makes a specific, verifiable assertion
           - Is supported by concrete evidence
           - Represents findings, contributions, or methodological advantages
           - Can be from any section except abstract

        2. For each identified claim:
           - Extract ALL supporting or contradicting evidence (experimental results, data, or methodology)
           - Evaluate the evidence strength and limitations
           - Assess how well conclusions align with evidence

        Return ONLY the following JSON structure:
        {{
            "analysis": [
                {{
                    "claim_id": number,
                    "claim": {{
                        "text": "statement of the claim",
                        "type": "methodology/result/contribution/performance",
                        "location": "section/paragraph",
                        "exact_quote": "verbatim text from paper"
                    }},
                    "evidence": [
                        {{
                            "evidence_text": "specific experimental result/data",
                            "strength": "strong/moderate/weak",
                            "limitations": "specific limitations",
                            "location": "section/paragraph",
                            "exact_quote": "verbatim text from paper"
                        }}
                    ],
                    "evaluation": {{
                        "conclusion_justified": true/false,
                        "robustness": "high/medium/low",
                        "justification": "explanation of evidence-conclusion alignment",
                        "key_limitations": "critical limitations affecting validity",
                        "confidence_level": "high/medium/low"
                    }}
                }}
            ]
        }}

        Ensure:
        - ALL substantive claims are captured
        - Evaluations are objective and well-reasoned
        - All locations and quotes are precise
        - Multiple pieces of evidence per claim are included when present
        """
        
        response = self._get_model_response(comprehensive_prompt)
        self.execution_times["single_pass_analysis"] = time.time() - start_time

        return self._parse_json_response(response)


    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response and handle errors"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON content found in response")
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing response: {e}")
            print("Raw response:", response)
            return None

    def combine_results(self, analysis_results: Dict) -> tuple:
        """Restructure the single-pass analysis results into the desired format"""
        claims = {
            "claims": [
                {
                    "claim_id": item["claim_id"],
                    "claim_text": item["claim"]["text"],
                    "location": item["claim"]["location"],
                    "claim_type": item["claim"]["type"],
                    "exact_quote": item["claim"]["exact_quote"]
                }
                for item in analysis_results["analysis"]
            ]
        }
        
        evidence_results = [
            {
                "claim_id": item["claim_id"],
                "evidence": [
                    {
                        "evidence_id": idx + 1,
                        "evidence_text": ev["evidence_text"],
                        "evidence_type": "primary",
                        "strength": ev["strength"],
                        "limitations": ev["limitations"],
                        "location": ev["location"],
                        "exact_quote": ev["exact_quote"]
                    }
                    for idx, ev in enumerate(item["evidence"])
                ]
            }
            for item in analysis_results["analysis"]
        ]
        
        conclusions = {
            "conclusions": [
                {
                    "claim_id": item["claim_id"],
                    "author_conclusion": item["evaluation"]["justification"],
                    "conclusion_justified": item["evaluation"]["conclusion_justified"],
                    "robustness_analysis": item["evaluation"]["robustness"],
                    "limitations": item["evaluation"]["key_limitations"],
                    "evidence_alignment": item["evaluation"]["justification"],
                    "confidence_level": item["evaluation"]["confidence_level"]
                }
                for item in analysis_results["analysis"]
            ],
            "analysis_metadata": {
                "total_claims_analyzed": len(analysis_results["analysis"]),
                "claims_with_conclusions": len(analysis_results["analysis"]),
                "analysis_timestamp": str(datetime.datetime.now())
            }
        }
        
        final_results = {
            "paper_analysis": []
        }
        
        for item in analysis_results["analysis"]:
            claim_id = item["claim_id"]
            analysis = {
                "claim_id": claim_id,
                "claim": item["claim"]["text"],
                "claim_location": item["claim"]["location"],
                "evidence": item["evidence"],
                "evidence_locations": [ev["location"] for ev in item["evidence"]],
                "conclusion": {
                    "author_conclusion": item["evaluation"]["justification"],
                    "conclusion_justified": item["evaluation"]["conclusion_justified"],
                    "robustness_analysis": item["evaluation"]["robustness"],
                    "limitations": item["evaluation"]["key_limitations"],
                    "conclusion_location": item["claim"]["location"]
                }
            }
            final_results["paper_analysis"].append(analysis)
        
        final_results["execution_times"] = {
            "single_pass_analysis_time": f"{self.execution_times['single_pass_analysis']:.2f} seconds",
            "total_execution_time": f"{self.execution_times['total_time']:.2f} seconds"
        }
        
        return claims, evidence_results, conclusions, final_results

    def print_analysis_results(self, final_results: Dict):
        """Print the analysis results in a readable format"""
        print("\n=== Complete Paper Analysis ===\n")
        
        for analysis in final_results['paper_analysis']:
            print(f"Claim {analysis['claim_id']}:")
            print(f"Statement: {analysis['claim']}")
            print("\nEvidence:")
            for evidence in analysis['evidence']:
                print(f"- {evidence['evidence_text']}")
                print(f"  Strength: {evidence['strength']}")
                print(f"  Limitations: {evidence['limitations']}")
            
            print("\nConclusion:")
            print(f"Author's Conclusion: {analysis['conclusion']['author_conclusion']}")
            print(f"Justified by Evidence: {'Yes' if analysis['conclusion']['conclusion_justified'] else 'No'}")
            print(f"Robustness: {analysis['conclusion']['robustness_analysis']}")
            print(f"Limitations: {analysis['conclusion']['limitations']}")
            print("\n" + "-"*50 + "\n")

    def save_results(self, results: Dict, base_filename: str):
        """Save analysis results to files"""
        output_dir = Path(f'{self.model_choice}_single_pass')
        output_dir.mkdir(exist_ok=True)

        results["execution_times"] = {
            "single_pass_analysis_time": f"{self.execution_times['single_pass_analysis']:.2f} seconds",
            "total_execution_time": f"{self.execution_times['total_time']:.2f} seconds"
        }
        
        # Save full JSON results
        json_path = output_dir / f'{base_filename}_analysis.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        # Save readable text summary
        text_path = output_dir / f'{base_filename}_summary.txt'
        with open(text_path, 'w', encoding='utf-8') as f:
            for analysis in results['analysis']:
                f.write(f"Claim {analysis['claim_id']}:\n")
                f.write(f"Type: {analysis['claim']['type']}\n")
                f.write(f"Statement: {analysis['claim']['text']}\n")
                f.write(f"Location: {analysis['claim']['location']}\n")
                f.write(f"Exact Quote: {analysis['claim']['exact_quote']}\n\n")
                
                f.write("Evidence:\n")
                for evidence in analysis['evidence']:
                    f.write(f"- Evidence Text: {evidence['evidence_text']}\n")
                    f.write(f"  Strength: {evidence['strength']}\n")
                    f.write(f"  Location: {evidence['location']}\n")
                    f.write(f"  Limitations: {evidence['limitations']}\n")
                    f.write(f"  Exact Quote: {evidence['exact_quote']}\n\n")
                
                eval_data = analysis['evaluation']
                f.write("Evaluation:\n")
                f.write(f"Conclusion Justified: {'Yes' if eval_data['conclusion_justified'] else 'No'}\n")
                f.write(f"Robustness: {eval_data['robustness']}\n")
                f.write(f"Confidence Level: {eval_data['confidence_level']}\n")
                f.write(f"Justification: {eval_data['justification']}\n")
                f.write(f"Key Limitations: {eval_data['key_limitations']}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
        
        # Generate summary statistics
        stats_path = output_dir / f'{base_filename}_statistics.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            total_claims = len(results['analysis'])
            justified_claims = sum(1 for a in results['analysis'] 
                                 if a['evaluation']['conclusion_justified'])
            
            f.write("Analysis Statistics:\n")
            f.write(f"Total Claims Analyzed: {total_claims}\n")
            f.write(f"Justified Claims: {justified_claims}\n")
            
            # Evidence strength distribution
            strength_levels = {}
            for analysis in results['analysis']:
                for evidence in analysis['evidence']:
                    strength = evidence['strength']
                    strength_levels[strength] = strength_levels.get(strength, 0) + 1
            
            f.write("\nEvidence Strength Distribution:\n")
            total_evidence = sum(strength_levels.values())
            for strength, count in strength_levels.items():
                f.write(f"{strength}: {count} pieces ({count/total_evidence*100:.1f}%)\n")


def main():
    input_folder = 'Research_papers'
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]

    print(f"Using model: {SELECTED_MODEL}")
    
    # Initialize model once before processing any files
    ModelManager.initialize_model()

    for filename in pdf_files:
        basefile_name = Path(filename).stem
        try:
            filename = f"{input_folder}/{filename}"
            total_start_time = time.time()

            # Initialize analyzer (will use already loaded model)
            analyzer = SinglePassPaperAnalyzer()
            
            print(f"Starting analysis of {filename}")
            
            # Extract text from PDF
            print("Extracting text from PDF...")
            analyzer.extract_text_from_pdf(filename)
            
            # Perform single-pass analysis
            print("Analyzing paper...")
            analysis_results = analyzer.analyze_paper(filename)
            analyzer.execution_times["total_time"] = time.time() - total_start_time

            # Restructure results into desired format
            claims, evidence_results, conclusions, final_results = analyzer.combine_results(analysis_results)
            
            # Print results
            analyzer.print_analysis_results(final_results)
            
            # Save detailed results
            analyzer.save_results(analysis_results, basefile_name)
            
        except Exception as e:
            print(f"Error analyzing paper: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
