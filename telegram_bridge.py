"""
Telegram Bridge for Chronos Pipeline
Calls the existing full pipeline: OCR → KG → Pattern Discovery → Hypothesis Verification
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add chronos/app to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from pipeline import run_pipeline
from kg_pattern_discovery import KGPatternDiscovery
from hypothesis_verifier import HypothesisVerifier

# Load environment from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")


def process_telegram_image(image_path: str, user_id: str = "telegram_user") -> dict:
    """
    Process a Telegram image through the complete Chronos pipeline.

    Args:
        image_path: Path to the downloaded image file
        user_id: Telegram user ID (for tracking/logging)

    Returns:
        Dictionary with hypothesis verification results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    element_id = f"telegram_{user_id}_{timestamp}"

    # Output paths
    chronos_dir = Path(__file__).parent
    output_text_file = chronos_dir / "chronos_output" / f"{element_id}_text.txt"
    output_text_file.parent.mkdir(exist_ok=True)

    # Neo4j configuration
    NEO4J_URL = os.environ.get("NEO4J_URL", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "0123456789")

    print("\n" + "="*80)
    print("🚀 CHRONOS PIPELINE FOR TELEGRAM IMAGE")
    print("="*80)
    print(f"📷 Image: {Path(image_path).name}")
    print(f"👤 User: {user_id}")
    print(f"🆔 Element ID: {element_id}")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)

    try:
        # STEP 1-2: Run full pipeline (OCR + Knowledge Graph)
        print("\n" + "="*80)
        print("📋 RUNNING FULL PIPELINE (OCR + KNOWLEDGE GRAPH)")
        print("="*80)

        # For images, we need to use a different approach than PDFs
        # Use the pipeline directly instead of run_pipeline
        from pipeline import MedicalDocumentPipeline

        pipeline = MedicalDocumentPipeline(
            neo4j_url=NEO4J_URL,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD,
            use_advanced_ocr=True,
            use_advanced_kg=False  # Use GPT-4o-mini
        )

        # Process image with correct parameters
        ocr_config = {
            "use_preprocessing": True,
            "enhancement_level": "aggressive",
            "medical_context": True,
            "save_debug_images": False
        }

        extracted_text, graph_elements = pipeline.process_document(
            input_file=image_path,
            output_text_file=str(output_text_file),
            ocr_config=ocr_config,
            element_id=element_id,
            kg_chunk_size=10000,
            enable_chunking=True
        )

        print(f"\n✅ Pipeline complete - Extracted {len(extracted_text)} characters")

        # STEP 3: Pattern Discovery
        print("\n" + "="*80)
        print("🔍 STEP 3: PATTERN DISCOVERY")
        print("="*80)

        pattern_discovery = KGPatternDiscovery(
            neo4j_url=NEO4J_URL,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )

        patterns = pattern_discovery.discover_patterns(
            max_length=3,
            max_patterns_per_length=5
        )
        pattern_discovery.close()

        # Extract questions from patterns
        questions = [p['question'] for p in patterns if p.get('question')]

        if not questions:
            print("\n⚠️  No questions generated from patterns")
            return {
                "success": False,
                "error": "No hypotheses generated from the image",
                "ocr_text": extracted_text
            }

        print(f"\n✅ Generated {len(questions)} questions from patterns")

        # STEP 4: Hypothesis Verification
        print("\n" + "="*80)
        print("🔬 STEP 4: HYPOTHESIS VERIFICATION")
        print("="*80)

        verifier = HypothesisVerifier(output_dir=str(chronos_dir / "hypothesis_results"))
        results = verifier.verify_questions_sync(questions)

        print("\n" + "="*80)
        print("✅ CHRONOS PIPELINE COMPLETE")
        print("="*80)
        print(f"⏰ Finished: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Summary:")
        print(f"   - OCR Text: {len(extracted_text)} characters")
        print(f"   - Patterns: {len(patterns)}")
        print(f"   - Questions: {len(questions)}")
        print(f"   - Verifications: {len(results)}")

        return {
            "success": True,
            "hypotheses": questions,
            "verification_results": results,
            "ocr_text": extracted_text,
            "element_id": element_id
        }

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the bridge
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        if os.path.exists(test_image):
            result = process_telegram_image(test_image, "test_user")

            print("\n" + "="*80)
            print("RESULTS")
            print("="*80)
            print(f"Success: {result['success']}")
            if result.get('error'):
                print(f"Error: {result['error']}")
            else:
                print(f"\nHypotheses generated: {len(result['hypotheses'])}")
                for i, h in enumerate(result['hypotheses'], 1):
                    print(f"  {i}. {h}")

                print(f"\nVerification results: {len(result['verification_results'])}")
        else:
            print(f"❌ Image not found: {test_image}")
    else:
        print("Usage: python telegram_bridge.py <image_path>")
