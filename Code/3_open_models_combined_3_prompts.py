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


import os
from huggingface_hub import login

# Method 1: Set token directly in code
os.environ["HUGGING_FACE_HUB_TOKEN"] = "Hugging_Face_Token"

# Global variable for model selection
SELECTED_MODEL = "llama"  # Change this to "llama" or "phi"

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

class PaperAnalyzer:
    def __init__(self):
        self.paper_text = None
        self.execution_times = {
            "claims_analysis": 0,
            "evidence_analysis": 0,
            "conclusions_analysis": 0,
            "total_time": 0
        }
        self.model_choice = SELECTED_MODEL
        # Get the already initialized model
        self.model, self.sampling_params = ModelManager.get_model()

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


    def extract_text_from_pdf(self, filename: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        try:
            self.paper_text = pymupdf4llm.to_markdown(filename)
            return self.paper_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def get_all_claims(self, filename: str) -> Dict:
        """Get all claims in one pass"""
        try:
            if not self.paper_text:
                text = self.extract_text_from_pdf(filename)
            else:
                text = self.paper_text

            print(f"Processing file: {filename}")
            start_time = time.time()

            claims_prompt = f"""
            paper text: {text}
            task is to identify all statements in the text that meet the following criteria for a claim:
            1. Makes a specific, testable assertion about results, methods, or contributions
            2. Represents a novel finding, improvement, or advancement
            3. Presents a clear position or conclusion

            Make sure to:
            1. Include both major and minor claims
            2. Don't miss any claims
            3. Present each claim as a separate item
            
            Return ONLY the following JSON structure:
            {{
                "claims": [
                    {{
                        "claim_id": 1,
                        "claim_text": "statement of the claim",
                        "location": "section/paragraph where this claim appears",
                        "claim_type": "Nature of the claim",
                        "exact_quote": "complete verbatim text containing the claim"
                    }}
                ]
            }}
            """
            
            response = self._get_model_response(claims_prompt)
            result = self._parse_json_response(response)
            self.execution_times["claims_analysis"] = time.time() - start_time

            print("Claims extraction completed")
            return result
        except Exception as e:
            print(f"Error in get_all_claims: {str(e)}")
            raise

    def get_all_evidence(self, filename: str, claims: Dict) -> Dict:
        """Get evidence for all claims in one pass"""
        try:
            start_time = time.time()

            if not self.paper_text:
                text = self.extract_text_from_pdf(filename)
            else:
                text = self.paper_text
            
            claims_text = "\n".join([f"Claim {c['claim_id']}: {c['claim_text']}" 
                                   for c in claims['claims']])
            print("Processing evidence for claims:", claims_text)
            
            evidence_prompt = f"""
            Paper text: {text}

            For these claims:
            {claims_text}

             Please identify relevant evidence that:
            1. Directly supports or contradicts the claim's specific assertion
            2. Is presented with experimental results, data, or concrete examples
            3. Can be traced to specific methods, results, or discussion sections
            4. Is not from the abstract or introduction

            Return ONLY the following JSON:
            {{
                "evidence_sets": [
                    {{
                        "claim_id": number,
                        "evidence": [
                            {{
                                "evidence_id": number,
                                "evidence_text": "specific evidence",
                                "strength": "strong/moderate/weak",
                                "limitations": "key limitations",
                                "location": "section/paragraph",
                                "exact_quote": "verbatim text"
                            }}
                        ]
                    }}
                ]
            }}
            """
            
            response = self._get_model_response(evidence_prompt)
            result = self._parse_json_response(response)
            self.execution_times["evidence_analysis"] = time.time() - start_time

            print("Evidence extraction completed")
            return result
        except Exception as e:
            print(f"Error in get_all_evidence: {str(e)}")
            raise

    def get_all_conclusions(self, filename: str, claims: Dict, evidence_sets: Dict) -> Dict:
        """Analyze conclusions for all claims and evidence in one pass"""
        try:
            if not self.paper_text:
                text = self.extract_text_from_pdf(filename)
            else:
                text = self.paper_text
            start_time = time.time()
            
            analysis_summary = []
            for claim in claims['claims']:
                claim_id = claim['claim_id']
                claim_evidence = next((e['evidence'] for e in evidence_sets['evidence_sets'] 
                                    if e['claim_id'] == claim_id), [])
                
                summary = f"\nClaim {claim_id}: {claim['claim_text']}\n"
                summary += "Evidence:\n"
                for evidence in claim_evidence:
                    summary += f"- {evidence['evidence_text']}\n"
                analysis_summary.append(summary)
            
            analysis_text = "\n".join(analysis_summary)
            
            conclusions_prompt = f"""
            Paper text: {text}

            Analyze these claims and their evidence:
            {analysis_text}

            For each claim-evidence pair, evaluate:
            1. Whether the evidence justifies the claim
            2. The overall strength of support
            3. Any important limitations

            Return ONLY the following JSON:
            {{
                "conclusions": [
                    {{
                        "claim_id": number,
                        "conclusion_justified": true/false,
                        "robustness": "high/medium/low",
                        "key_limitations": "specific limitations",
                        "confidence_level": "high/medium/low"
                    }}
                ]
            }}
            """
            
            response = self._get_model_response(conclusions_prompt)
            result = self._parse_json_response(response)
            self.execution_times["conclusions_analysis"] = time.time() - start_time
            print("Conclusions analysis completed")
            return result
            
        except Exception as e:
            print(f"Error in get_all_conclusions: {str(e)}")
            raise

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response with better error handling"""
        try:
            print("Parsing response...")
            print("Raw response:", response)
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON content found in response")
                
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            print("Successfully parsed JSON response")
            return result
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print("Raw response:", response)
            raise

    def analyze_paper(self, filename: str) -> Dict:
        """Complete paper analysis using three-prompt approach"""
        try:
            total_start_time = time.time()

            print("Extracting text from PDF...")
            self.extract_text_from_pdf(filename)

            print("Extracting claims...")
            claims = self.get_all_claims(filename)
            if not claims:
                raise Exception("Failed to extract claims")

            print("Extracting evidence...")
            evidence_sets = self.get_all_evidence(filename, claims)
            if not evidence_sets:
                raise Exception("Failed to extract evidence")

            print("Analyzing conclusions...")
            conclusions = self.get_all_conclusions(filename, claims, evidence_sets)
            if not conclusions:
                raise Exception("Failed to generate conclusions")

            self.execution_times["total_time"] = time.time() - total_start_time

            final_results = {
                "paper_analysis": []
            }

            for claim in claims['claims']:
                claim_id = claim['claim_id']
                
                evidence = next((e['evidence'] for e in evidence_sets['evidence_sets'] 
                            if e['claim_id'] == claim_id), [])
                
                conclusion = next((c for c in conclusions['conclusions'] 
                                if c['claim_id'] == claim_id), {})

                analysis_item = {
                    "claim_id": claim_id,
                    "claim": {
                        "text": claim['claim_text'],
                        "location": claim['location'],
                        "type": claim['claim_type'],
                        "exact_quote": claim['exact_quote']
                    },
                    "evidence": evidence,
                    "conclusion": {
                        "conclusion_justified": conclusion.get('conclusion_justified', False),
                        "robustness": conclusion.get('robustness', 'Not evaluated'),
                        "limitations": conclusion.get('key_limitations', 'Not specified'),
                        "confidence_level": conclusion.get('confidence_level', 'low')
                    }
                }
                
                final_results['paper_analysis'].append(analysis_item)

            final_results["execution_times"] = {
                "claims_analysis_time": f"{self.execution_times['claims_analysis']:.2f} seconds",
                "evidence_analysis_time": f"{self.execution_times['evidence_analysis']:.2f} seconds",
                "conclusions_analysis_time": f"{self.execution_times['conclusions_analysis']:.2f} seconds",
                "total_execution_time": f"{self.execution_times['total_time']:.2f} seconds"
            }

            return final_results

        except Exception as e:
            print(f"Error in paper analysis: {str(e)}")
            return None

    def save_results(self, results: Dict, filename: str):
        """Save analysis results in multiple formats"""
        try:
            base_filename = Path(filename).stem
            
            # Create output directory based on model choice
            output_dir = f"{self.model_choice}_analysis"
            os.makedirs(output_dir, exist_ok=True)

            # Save detailed JSON results
            json_filename = f'{output_dir}/{base_filename}_analysis.json'
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)

            # Save human-readable summary
            summary_filename = f'{output_dir}/{base_filename}_summary.txt'
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write("=== Paper Analysis Summary ===\n\n")
                
                for analysis in results['paper_analysis']:
                    f.write(f"Claim {analysis['claim_id']}:\n")
                    f.write(f"Statement: {analysis['claim']['text']}\n")
                    f.write(f"Location: {analysis['claim']['location']}\n")
                    f.write(f"Type: {analysis['claim']['type']}\n")
                    f.write(f"Quote: {analysis['claim']['exact_quote']}\n\n")
                    
                    f.write("Evidence:\n")
                    for evidence in analysis['evidence']:
                        f.write(f"- {evidence['evidence_text']}\n")
                        f.write(f"  Strength: {evidence['strength']}\n")
                        f.write(f"  Location: {evidence['location']}\n")
                        f.write(f"  Limitations: {evidence['limitations']}\n")
                        f.write(f"  Quote: {evidence['exact_quote']}\n\n")
                    
                    f.write("Conclusion:\n")
                    f.write(f"Justified: {analysis['conclusion']['conclusion_justified']}\n")
                    f.write(f"Robustness: {analysis['conclusion']['robustness']}\n")
                    f.write(f"Limitations: {analysis['conclusion']['limitations']}\n")
                    f.write(f"Confidence: {analysis['conclusion']['confidence_level']}\n")
                    f.write("\n" + "="*50 + "\n\n")

            # Save statistics
            stats_filename = f'{output_dir}/{base_filename}_stats.txt'
            with open(stats_filename, 'w', encoding='utf-8') as f:
                f.write("Analysis Statistics:\n\n")
                f.write(f"Total Claims Analyzed: {len(results['paper_analysis'])}\n")
                
                total_evidence = sum(len(analysis['evidence']) for analysis in results['paper_analysis'])
                f.write(f"Total Evidence Pieces: {total_evidence}\n")
                
                confidence_levels = {}
                for analysis in results['paper_analysis']:
                    level = analysis['conclusion']['confidence_level']
                    confidence_levels[level] = confidence_levels.get(level, 0) + 1
                
                f.write("\nConfidence Level Distribution:\n")
                for level, count in confidence_levels.items():
                    f.write(f"{level}: {count} claims\n")

                f.write("\nExecution Times:\n")
                f.write(f"Claims Analysis: {self.execution_times['claims_analysis']:.2f} seconds\n")
                f.write(f"Evidence Analysis: {self.execution_times['evidence_analysis']:.2f} seconds\n")
                f.write(f"Conclusions Analysis: {self.execution_times['conclusions_analysis']:.2f} seconds\n")
                f.write(f"Total Execution Time: {self.execution_times['total_time']:.2f} seconds\n")

            print(f"Results saved to {output_dir}/:")
            print(f"- Detailed analysis: {json_filename}")
            print(f"- Summary: {summary_filename}")
            print(f"- Statistics: {stats_filename}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")

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
            analyzer = PaperAnalyzer()
            
            print(f"Starting analysis of {filename}")
            results = analyzer.analyze_paper(filename)
            
            if results:
                analyzer.save_results(results, filename)
                print("Analysis completed successfully")
            else:
                print("Analysis failed to produce results")
            
        except Exception as e:
            print(f"Error in main execution: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
