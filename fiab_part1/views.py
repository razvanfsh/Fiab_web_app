# fiab_part1/views.py
import json
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .Fiab_main import calculate_reliability

logger = logging.getLogger(__name__)

def index(request):
    """Render the front page with CSRF token"""
    return render(request, 'fiab_part1/front_page.html')

@require_http_methods(["POST"])
def compute_reliability(request):
    """Handle reliability calculations with proper validation"""
    try:
        # Debug raw request data
        raw_data = request.body.decode('utf-8')
        logger.debug(f"Raw request data: {raw_data}")
        
        data = json.loads(raw_data)
        logger.debug(f"Parsed data: {data}")

        # Validate required fields
        required_fields = ['num_components', 'system_type', 'structures', 'p_value', 'lambda_value']
        for field in required_fields:
            if field not in data:
                return JsonResponse({'status': 'error', 'message': f'Missing field: {field}'}, status=400)

        # Convert and validate structures
        structures = []
        for idx, struct in enumerate(data['structures']):
            try:
                components = [int(c) for c in struct]
                if any(c < 1 or c > data['num_components'] for c in components):
                    raise ValueError(f"Component numbers must be between 1 and {data['num_components']}")
                structures.append(components)
            except (ValueError, TypeError) as e:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Invalid structure at position {idx+1}: {str(e)}'
                }, status=400)
            
        comp_index = data.get('comp_index', None)
        if comp_index is not None:
            # ensure it's an integer and valid
            comp_index = int(comp_index)
            if comp_index < 1 or comp_index > data['num_components']:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Invalid comp_index: must be between 1 and {data["num_components"]}'
                }, status=400)


        # Perform calculation
        results = calculate_reliability(
            n=data['num_components'],
            use_paths=data['system_type'] == 'paths',
            structures=structures,
            p_rob=data['p_value'],
            l=data['lambda_value'],
            comp_index=comp_index
        )

        # Add input parameters to results for display
        results.update({
            'num_components': data['num_components'],
            'system_type': data['system_type'],
            'p_value': data['p_value'],
            'lambda_value': data['lambda_value'],
            'structures': structures,
            'comp_index': comp_index
        })

        logger.debug("Calculation successful, returning results")
        return JsonResponse({'status': 'success', 'results': results})

    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON format'}, status=400)
        
    except Exception as e:
        logger.exception("Unexpected error in computation")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    