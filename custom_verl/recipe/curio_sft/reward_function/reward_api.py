#!/usr/bin/env python3

import os
import sys
import json
import traceback
import multiprocessing as mp
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from pebble import ProcessPool
from tqdm import tqdm
import time
import atexit
import sys
from math_verify import parse, verify
from custom_verl.recipe.curio_sft.reward_function.oat_math_grader import boxed_reward_fn as oat_evaluate

app = Flask(__name__)

process_pool = None

def labeling_responses(responses: list[str], golden_answer: str, timeout=False):
    if not timeout:
        responses = [parse(responses[i], parsing_timeout=None) for i in range(len(responses))]
        golden_answer = parse("$" + golden_answer + "$", parsing_timeout=None)
        labels = [verify(golden_answer, responses[i], timeout_seconds=None) for i in range(len(responses))]
    else:
        responses = [parse(responses[i], parsing_timeout=5) for i in range(len(responses))]
        golden_answer = parse("$" + golden_answer + "$", parsing_timeout=5)
        labels = [verify(golden_answer, responses[i], timeout_seconds=5) for i in range(len(responses))]
    return labels,responses


def process_single_item_worker(item_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        i = item_data['i']
        sequences_str = item_data['sequences_str']
        ground_truth = item_data['ground_truth']
        reward_impl_version = item_data['reward_impl_version']
        oat_grader = item_data.get('oat_grader', False)
        valid_response_length = item_data['valid_response_length']
        enable_timeout = item_data.get('enable_timeout', True)
        data_source = item_data.get('data_source', 'math')
        extra_info = item_data.get('extra_info', None)
        
        final_score = 0.0
        think_format = True if reward_impl_version else False
        
        if think_format:
            if  '</think>' in sequences_str:
                model_solution = sequences_str.split('</think>')[-1]
            else:
                return {
                    'i': i,
                    'score': 0.0,
                    'valid_response_length': valid_response_length,
                    'error': None,
                    'extracted_answers': ''
                }
        else:
            model_solution = sequences_str
        
        try:
            labels,extracted_answers = labeling_responses([model_solution], ground_truth, timeout=enable_timeout)
                      
            if labels and labels[0]:
                final_score = 1.0
        except Exception as e:
            final_score = 0.0
        
        if oat_grader and not data_source.startswith('codegen'):
            try:
                oat_score = oat_evaluate(sequences_str, ground_truth, fast=False)[1]
                final_score = max(float(oat_score), float(final_score))
            except Exception as e:
                pass
        
        return {
            'i': i,
            'score': final_score,
            'valid_response_length': valid_response_length,
            'error': None,
            'extracted_answers': str(extracted_answers[0])
        }
        
    except Exception as e:
        return {
            'i': item_data.get('i', -1),
            'score': 0.0,
            'valid_response_length': item_data.get('valid_response_length', 0),
            'error': str(e),
            'extracted_answers': ''
        }

def init_process_pool(max_workers: int):
    global process_pool
    
    if process_pool is not None:
        process_pool.close()
        process_pool.join()
    
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    process_pool = ProcessPool(max_workers=max_workers)

def cleanup_process_pool():
    global process_pool
    if process_pool is not None:
        process_pool.close()
        process_pool.join()
        process_pool = None

atexit.register(cleanup_process_pool)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '2.0.0',
        'pool_type': 'ProcessPool'
    })

@app.route('/process_single', methods=['POST'])
def process_single():
    global process_pool
    
    try:
        data = request.get_json()
        
        if process_pool is None:
            return jsonify({'error': 'Process pool not initialized'}), 500
        
        try:
            future = process_pool.schedule(process_single_item_worker, args=[data], timeout=10)
            result = future.result()
            print('process_single', result)
            return jsonify(result)
        except TimeoutError:
            return jsonify({
                'i': data.get('i', -1),
                'score': 0.0,
                'valid_response_length': data.get('valid_response_length', 0),
                'error': 'Task timeout (30s)',
                'extracted_answers': ''
            })
        except Exception as e:
            return jsonify({
                'i': data.get('i', -1),
                'score': 0.0,
                'valid_response_length': data.get('valid_response_length', 0),
                'error': str(e),
                'extracted_answers': ''
            })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/process_batch', methods=['POST'])
def process_batch():
    global process_pool
    
    try:
        data = request.get_json()
        items = data.get('items', [])
        data_sources_count = {}
        for item in items:
            data_source = item.get('data_source', 'None')
            if data_source not in data_sources_count:
                data_sources_count[data_source] = 0
            data_sources_count[data_source] += 1
        print(f"data_sources_count: {data_sources_count}")
        
        if not items:
            return jsonify({'error': 'No items provided'}), 400
        
        if process_pool is None:
            return jsonify({'error': 'Process pool not initialized'}), 500
        
        results = []
        timeout_cnt = 0
        
        try:
            futures = {}
            for item in items:
                future = process_pool.schedule(process_single_item_worker, args=[item], timeout=10)
                futures[future] = item
            
            results = []
            with tqdm(total=len(items), desc="Processing batch") as progress_bar:
                for future, item in futures.items():
                    try:
                        result = future.result()
                        results.append(result)
                    except TimeoutError:
                        results.append({
                            'i': item['i'],
                            'score': 0.0,
                            'valid_response_length': item['valid_response_length'],
                            'error': 'Task timeout',
                            'extracted_answers': ''
                        })
                        timeout_cnt += 1
                        future.cancel() 
                    except Exception as e:
                        results.append({
                            'i': item['i'],
                            'score': 0.0,
                            'valid_response_length': item['valid_response_length'],
                            'error': str(e),
                            'extracted_answers': ''
                        })
                        future.cancel() 
                    progress_bar.update(1)
        
        except Exception as e:
            return jsonify({
                'error': f'Process pool error: {str(e)}',
                'traceback': traceback.format_exc()
            }), 500
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timeout_count': timeout_cnt
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/pool_status', methods=['GET'])
def pool_status():
    global process_pool
    
    if process_pool is None:
        return jsonify({
            'status': 'not_initialized',
            'pool': None
        })
    
    return jsonify({
        'status': 'active',
        'pool_type': 'ProcessPool',
        'max_workers': getattr(process_pool, '_max_workers', 'unknown')
    })

@app.route('/reinit_pool', methods=['POST'])
def reinit_pool():
    try:
        data = request.get_json() or {}
        max_workers = data.get('max_workers', 32)
        
        init_process_pool(max_workers)
        
        return jsonify({
            'status': 'success',
            'message': f'Process pool reinitialized with {max_workers} workers'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Reward Calculation Server with ProcessPool')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=19876, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker processes')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    init_process_pool(args.workers)
    
    print(f"Starting Reward Server on {args.host}:{args.port}")
    print(f"Worker processes: {args.workers}")
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        pass 
    finally:
        cleanup_process_pool() 
