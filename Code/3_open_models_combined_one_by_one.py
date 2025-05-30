import json
import datetime
import pymupdf4llm
from typing import Dict, List, Any
import time
from pathlib import Path
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from vllm import LLM, SamplingParams
import traceback
import re


os.environ["HUGGING_FACE_HUB_TOKEN"] = "Hugging_Face_Token"

# Global model selection
SELECTED_MODEL = "phi"  # Options: "mistral", "llama", "phi"

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
                    "max_new_tokens": 8192,
                    "do_sample": True,
                    "temperature": 0.01,
                    "return_full_text": False
                }
                prompt_text = f"System: You are a helpful assistant specialized in analyzing research papers.\nUser: {prompt}"
                output = self.model(prompt_text, **generation_args)
                response = output[0]['generated_text']
            
            return response
            
        except Exception as e:
            print(f"Error generating response with {self.model_choice}: {str(e)}")
            return ""

    def get_claims(self, filename: str) -> Dict:
        """Extract all claims from the paper"""
        if not self.paper_text:
            text = self.extract_text_from_pdf(filename)
        else:
            text = self.paper_text
        
        if not self.paper_text:
            raise Exception("Failed to extract text from PDF")
        start_time = time.time()

        claims_prompt = f"""
        Analyze this research paper and extract ALL possible claims made by the authors.
        Paper text: {text}
        
        Your task is to identify all statements in the text that meet the following criteria for a claim:
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
        self.execution_times["claims_analysis"] = time.time() - start_time

        return self._parse_json_response(response)

    def analyze_evidence(self, filename: str, claims: Dict) -> List[Dict]:
        """Find evidence for each claim"""
        if not claims or not isinstance(claims, dict) or 'claims' not in claims:
            print("No valid claims provided for evidence analysis")
            return []

        if not self.paper_text:
            text = self.extract_text_from_pdf(filename)
        else:
            text = self.paper_text
        
        start_time = time.time()
        evidence_results = []
        
        for claim in claims.get('claims', []):
            try:
                claim_id = claim.get('claim_id')
                if claim_id is None:
                    continue
                    
                evidence_prompt = f"""
                Paper text: {text}
                
                For the following claim from the paper:
                "{claim['claim_text']}"
                
                Please identify relevant evidence that:
                1. Directly supports or contradicts the claim's specific assertion
                2. Is presented with experimental results, data, or concrete examples
                3. Can be traced to specific methods, results, or discussion sections
                4. Is not from the abstract or introduction

                Return ONLY the following JSON structure:
                {{
                    "claim_id": {claim_id},
                    "evidence": [
                        {{
                            "evidence_id": 1,
                            "evidence_text": "specific experimental result/data point",
                            "evidence_type": "primary/secondary",
                            "strength": "strong/moderate/weak",
                            "limitations": "stated limitations or assumptions",
                            "location": "specific section & paragraph",
                            "exact_quote": "verbatim text from paper"
                        }}
                    ]
                }}
                """

                response = self._get_model_response(evidence_prompt)
                result = self._parse_json_response(response)
                
                if result and isinstance(result, dict):
                    if 'claim_id' not in result:
                        result['claim_id'] = claim_id
                    evidence_results.append(result)
                
            except Exception as e:
                print(f"Error analyzing evidence for claim {claim.get('claim_id', 'unknown')}: {str(e)}")
                continue
                    
        self.execution_times["evidence_analysis"] = time.time() - start_time
        return evidence_results

    def analyze_conclusions(self, filename: str, claims: Dict, evidence_results: List[Dict]) -> Dict:
        """Analyze conclusions by processing each claim-evidence pair individually"""
        if not self.paper_text:
            text = self.extract_text_from_pdf(filename)
        else:
            text = self.paper_text

        start_time = time.time()
        all_conclusions = []
        claims_list = claims.get('claims', [])

        def build_evidence_summary(claim_id):
            """Helper function to build evidence summary for a single claim"""
            claim_evidence = next((e['evidence'] for e in evidence_results if e.get('claim_id') == claim_id), [])
            evidence_text = []
            for idx, evidence in enumerate(claim_evidence, 1):
                evidence_text.append(
                    f"  Evidence {idx}:\n"
                    f"    - Text: {evidence.get('evidence_text', 'No text provided')}\n"
                    f"    - Strength: {evidence.get('strength', 'Not specified')}\n"
                    f"    - Limitations: {evidence.get('limitations', 'None specified')}\n"
                    f"    - Location: {evidence.get('location', 'Location not specified')}"
                )
            return "\n".join(evidence_text)

        for claim in claims_list:
            claim_id = claim.get('claim_id')
            print(f"\nAnalyzing conclusion for Claim {claim_id}...")

            single_claim_analysis = f"""
            Claim {claim_id}:
            Statement: {claim.get('claim_text', 'No text provided')}
            Location: {claim.get('location', 'Location not specified')}
            
            Evidence Summary:
            {build_evidence_summary(claim_id)}
            """

            single_conclusion_prompt = f"""
            Paper text: {text}
            
            Analyze the following claim and its supporting evidence:
            {single_claim_analysis}

            Provide a comprehensive conclusion analysis following these guidelines:

            1. Evidence Assessment:
            - Evaluate the strength and quality of ALL evidence presented
            - Consider both supporting and contradicting evidence
            - Assess the methodology and reliability of evidence

            2. Conclusion Analysis:
            - Determine what the authors concluded about this specific claim
            - Evaluate if the conclusion is justified by the evidence
            - Consider the relationship between evidence quality and conclusion strength

            3. Robustness Evaluation:
            - Assess how well the evidence supports the conclusion
            - Consider methodological strengths and weaknesses
            - Evaluate the consistency of evidence

            4. Limitations Analysis:
            - Identify specific limitations in both evidence and conclusion
            - Consider gaps in methodology or data
            - Note any potential biases or confounding factors

            Return ONLY the following JSON structure:
            {{
                "conclusions": [
                    {{
                        "claim_id": {claim_id},
                        "author_conclusion": "detailed description of authors' conclusion based on evidence",
                        "conclusion_justified": true/false,
                        "justification_explanation": "detailed explanation of why conclusion is/isn't justified",
                        "robustness_analysis": "comprehensive analysis of evidence strength and reliability",
                        "limitations": "specific limitations and caveats",
                        "location": "section/paragraph where conclusion appears",
                        "evidence_alignment": "analysis of how well evidence aligns with conclusion",
                        "confidence_level": "high/medium/low based on evidence quality"
                    }}
                ]
            }}
            """

            try:
                response = self._get_model_response(single_conclusion_prompt)
                result = self._parse_json_response(response)

                if result and isinstance(result, dict) and 'conclusions' in result and result['conclusions']:
                    conclusion = result['conclusions'][0]
                    if conclusion.get('claim_id') == claim_id:
                        all_conclusions.append(conclusion)
                    else:
                        raise ValueError(f"Mismatched claim_id in response for claim {claim_id}")
                else:
                    raise ValueError(f"Invalid response format for claim {claim_id}")

            except Exception as e:
                print(f"Error analyzing conclusion for claim {claim_id}: {str(e)}")
                all_conclusions.append({
                    "claim_id": claim_id,
                    "author_conclusion": "No conclusion available",
                    "conclusion_justified": False,
                    "justification_explanation": "Analysis not available",
                    "robustness_analysis": "No robustness analysis available",
                    "limitations": "No limitations analysis available",
                    "location": "Location not specified",
                    "evidence_alignment": "No alignment analysis available",
                    "confidence_level": "low"
                })

        self.execution_times["conclusions_analysis"] = time.time() - start_time

        return {
            "conclusions": all_conclusions,
            "analysis_metadata": {
                "total_claims_analyzed": len(claims_list),
                "claims_with_conclusions": len(all_conclusions),
                "analysis_timestamp": str(datetime.datetime.now())
            }
        }

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response and handle errors more robustly"""
        try:
            # First try to find JSON structure with multiple start/end points
            json_candidates = []
            depth = 0
            start_idx = -1
            
            for i, char in enumerate(response):
                if char == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start_idx != -1:
                        json_str = response[start_idx:i+1]
                        try:
                            parsed = json.loads(json_str)
                            if 'claims' in parsed or 'evidence' in parsed or 'conclusions' in parsed:
                                return parsed
                            json_candidates.append(parsed)
                        except:
                            continue

            # If no valid JSON found with primary keys, try the first valid JSON
            if json_candidates:
                return json_candidates[0]

            # If still no valid JSON, try cleaning the response
            cleaned_response = response.strip()
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = cleaned_response[start_idx:end_idx]
                try:
                    return json.loads(json_str)
                except:
                    # Try to fix common JSON formatting issues
                    json_str = json_str.replace('\n', ' ').replace('\r', '')
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    return json.loads(json_str)

            raise ValueError("No valid JSON structure found in response")
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print("Raw response:", response)
            
            # Return a default structure based on the context
            if "claim" in response.lower():
                return {"claims": []}
            elif "evidence" in response.lower():
                return {"claim_id": None, "evidence": []}
            elif "conclusion" in response.lower():
                return {"conclusions": []}
            return None

    def combine_results(self, claims: Dict, evidence_results: List[Dict], conclusions: Dict) -> Dict:
        """Combine all analysis results into a final structured format"""
        final_results = {
            "paper_analysis": []
        }
        
        # Handle None or invalid inputs
        if not claims or not isinstance(claims, dict) or 'claims' not in claims:
            print("Warning: No valid claims provided")
            return final_results
        
        conclusions_dict = {
            c['claim_id']: c 
            for c in conclusions.get('conclusions', [])
        } if conclusions else {}
        
        # More robust evidence dictionary creation
        evidence_dict = {}
        for e in evidence_results:
            if isinstance(e, dict):
                claim_id = e.get('claim_id')
                if claim_id is not None:  # Only add if claim_id exists
                    evidence_dict[claim_id] = e.get('evidence', [])
        
        for claim in claims.get('claims', []):
            try:
                claim_id = claim['claim_id']
                conclusion = conclusions_dict.get(claim_id, {})
                evidence = evidence_dict.get(claim_id, [])
                
                analysis = {
                    "claim_id": claim_id,
                    "claim": claim.get('claim_text', ''),
                    "claim_location": claim.get('location', 'Location not specified'),
                    "evidence": evidence,
                    "evidence_locations": [ev.get('location', 'Location not specified') for ev in evidence],
                    "conclusion": {
                        "author_conclusion": conclusion.get('author_conclusion', 'No conclusion available'),
                        "conclusion_justified": conclusion.get('conclusion_justified', False),
                        "robustness_analysis": conclusion.get('robustness_analysis', 'No robustness analysis available'),
                        "limitations": conclusion.get('limitations', 'No limitations analysis available'),
                        "conclusion_location": conclusion.get('location', 'Location not specified')
                    }
                }
                
                final_results['paper_analysis'].append(analysis)
            except Exception as e:
                print(f"Error processing claim {claim.get('claim_id', 'unknown')}: {str(e)}")
                continue

        final_results["execution_times"] = {
            "claims_analysis_time": f"{self.execution_times['claims_analysis']:.2f} seconds",
            "evidence_analysis_time": f"{self.execution_times['evidence_analysis']:.2f} seconds",
            "conclusions_analysis_time": f"{self.execution_times['conclusions_analysis']:.2f} seconds",
            "total_execution_time": f"{self.execution_times['total_time']:.2f} seconds"
        }
        
        return final_results

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

            analyzer = PaperAnalyzer()
            
            print(f"Starting analysis of {filename}")
            
            if not analyzer.extract_text_from_pdf(filename):
                print(f"Failed to extract text from {filename}, skipping...")
                continue
            
            claims = analyzer.get_claims(filename)
            if not claims or not isinstance(claims, dict) or 'claims' not in claims:
                print(f"No valid claims found in {filename}, skipping...")
                continue

            evidence_results = analyzer.analyze_evidence(filename, claims)
            if not evidence_results:
                print("Warning: No evidence results generated")
                evidence_results = []

            conclusions = analyzer.analyze_conclusions(filename, claims, evidence_results)
            if not conclusions:
                print("Warning: No conclusions generated")
                conclusions = {"conclusions": []}

            analyzer.execution_times["total_time"] = time.time() - total_start_time
            final_results = analyzer.combine_results(claims, evidence_results, conclusions)
            
            # Print results
            analyzer.print_analysis_results(final_results)
            
            # Create output directory based on model
            output_dir = f'{SELECTED_MODEL}_one_by_one_analysis'
            os.makedirs(output_dir, exist_ok=True)

            # Save results
            with open(f'{output_dir}/{basefile_name}_analysis.json', 'w') as f:
                json.dump(final_results, f, indent=4)
            
            # Save intermediate results
            intermediate_results = {
                "claims": claims,
                "evidence": evidence_results,
                "conclusions": conclusions,
                "execution_times": final_results["execution_times"]
            }
            with open(f'{output_dir}/{basefile_name}_intermediate.json', 'w') as f:
                json.dump(intermediate_results, f, indent=4)
                

        except Exception as e:
            print(f"Error analyzing paper: {str(e)}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
