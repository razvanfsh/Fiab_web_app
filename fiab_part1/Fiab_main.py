import sympy

# ------------------------------
# Core Reliability Functions (Reusable)
# ------------------------------

def brute_force_reliability(n, use_paths, structures,comp_index=None):
    """Modified to remove hardcoded p_rob and l since they're not needed for structural calculations"""
    # ... (keep the existing brute_force_reliability implementation from your code) ...

    """
        Enumerates all 2^n states.
        Returns a dict with:
            - 'reliability_poly': system's multi-variate polynomial,
            - 'p_syms': the symbolic variables [p1..p_n],
            - 'success_count','fail_count',
            - scenario expressions & counts if comp_index is not None.
        """

    if structures is None:
        structures = []

    # Create p1..p_n
    p_syms = sympy.symbols(' '.join([f'p{i+1}' for i in range(n)]),
                            real=True, nonnegative=True)

    total_expr = 0
    success_count = 0

    # If we want scenario data for a specific component
    if comp_index is not None:
        scenario_expr = {
            "system_up_comp_up": 0,
            "system_up_comp_down": 0,
            "system_down_comp_up": 0,
            "system_down_comp_down": 0
        }
        scenario_count = {
            "system_up_comp_up": 0,
            "system_up_comp_down": 0,
            "system_down_comp_up": 0,
            "system_down_comp_down": 0
        }
    else:
        scenario_expr = None
        scenario_count = None

    # Enumerate all states
    for state_int in range(2**n):
        bits = [((state_int >> i) & 1) for i in range(n)]

        # Determine if system is up
        if use_paths:
            system_up = False
            for path in structures:
                if all(bits[c - 1] == 1 for c in path):
                    system_up = True
                    break
        else:
            # minimal cuts => system fails if any cut is fully down
            system_up = True
            for cut in structures:
                if all(bits[c - 1] == 0 for c in cut):
                    system_up = False
                    break

        if system_up:
            # Probability term for system up
            term = 1
            for i_comp in range(n):
                if bits[i_comp] == 1:
                    term *= p_syms[i_comp]
                else:
                    term *= (1 - p_syms[i_comp])
            total_expr += term
            success_count += 1

        # If we want scenario data for component comp_index
        if comp_index is not None:
            # Probability term for this state
            prob_term = 1
            for i_comp in range(n):
                if bits[i_comp] == 1:
                    prob_term *= p_syms[i_comp]
                else:
                    prob_term *= (1 - p_syms[i_comp])

            comp_up = (bits[comp_index - 1] == 1)
            if system_up and comp_up:
                scenario_expr["system_up_comp_up"] += prob_term
                scenario_count["system_up_comp_up"] += 1
            elif system_up and (not comp_up):
                scenario_expr["system_up_comp_down"] += prob_term
                scenario_count["system_up_comp_down"] += 1
            elif (not system_up) and comp_up:
                scenario_expr["system_down_comp_up"] += prob_term
                scenario_count["system_down_comp_up"] += 1
            else:
                scenario_expr["system_down_comp_down"] += prob_term
                scenario_count["system_down_comp_down"] += 1

    fail_count = 2**n - success_count
    reliability_poly = sympy.expand(total_expr)

    # Build the return dictionary
    result = {
        'reliability_poly': reliability_poly,
        'p_syms': p_syms,
        'success_count': success_count,
        'fail_count': fail_count
    }

    # If we have scenario data
    if comp_index is not None:
        result['system_up_comp_up_expr'] = scenario_expr["system_up_comp_up"]
        result['system_up_comp_up_count'] = scenario_count["system_up_comp_up"]
        result['system_up_comp_down_expr'] = scenario_expr["system_up_comp_down"]
        result['system_up_comp_down_count'] = scenario_count["system_up_comp_down"]
        result['system_down_comp_up_expr'] = scenario_expr["system_down_comp_up"]
        result['system_down_comp_up_count'] = scenario_count["system_down_comp_up"]
        result['system_down_comp_down_expr'] = scenario_expr["system_down_comp_down"]
        result['system_down_comp_down_count'] = scenario_count["system_down_comp_down"]
    return result
    

def collapse_to_p(expression, p_syms):
    # ... (keep existing implementation) ...
        """
        Substitutes p1..p_n => single symbolic p, returns (p, expanded_expr).
        """
        p = sympy.Symbol('p', real=True, nonnegative=True)
        single_p_expr = expression.subs({p_syms[i]: p for i in range(len(p_syms))})
        expanded_expr = sympy.expand(single_p_expr)
        return p, expanded_expr    

def compute_birnbaum_multivariable(reliability_poly, p_syms):
    # ... (keep existing implementation) ...
      """
      Compute partial derivatives (Birnbaum importance) for each p_i in
      the multi-variable polynomial.
      Returns a list of Sympy expressions [dR/dp1, dR/dp2, ..., dR/dpn].
      """
      birnbaum_list = []
      for psym in p_syms:
        partial_expr = sympy.diff(reliability_poly, psym)
        partial_expr_expanded = sympy.expand(partial_expr)
        birnbaum_list.append(partial_expr_expanded)
      return birnbaum_list


def compute_nus_derivative(single_p_expr):
    # ... (keep existing implementation) ...
        """
        Suppose R(p) is a single-variable reliability polynomial in p.
        We define nu_s(p) = p * dR/dp (not multiplied by lambda yet).
        Return the symbolic expression for nu_s(p).
        """
        p = sympy.Symbol('p', real=True, nonnegative=True)
        derivative = sympy.diff(single_p_expr, p)
        nu_s = p * derivative
        return nu_s  # The user can multiply by l or evaluate numeric as needed


def compute_structural_birnbaum(n, use_paths, structures):
    # ... (keep existing implementation) ...
      """
      Structural Birnbaum for each component i is:
        B_i = # of states where flipping i from 0->1 changes system from fail->up,
      or equivalently the difference in the structural function:
        phi(..., i=1, ...) - phi(..., i=0, ...).

      We'll do a brute force approach on the 'structural function' (0/1),
      ignoring probabilities, just to measure how many states i is 'critical'.

      Return: A list [B1, B2, ..., Bn] with the structural Birnbaum counts.
      """
      B_counts = [0]*n

      # We'll fix the other n-1 bits, and see how system changes from i=0 to i=1
        #   The system is up if use_paths => any path is fully up
        #                    use_cuts  => all cuts are not fully down
        # We'll reuse logic but ignoring probabilities, just boolean checks

      def system_up_bool(bits):
        #bits e lungimea n
        if use_paths:
          for path in structures:
            if all(bits[c-1]==1 for c in path):
              return True
          return False

        else:
          for cut in structures:
            if all(bits[c-1]==0 for c in cut):
              return False
          return True

        #Enumerate all states for the other bits
        for state_int in range(2**(n-1)):
          bits_base = [0]*n

          # We'll interpret state_int as the 'other n-1 bits' ignoring 1 comp at a time
            # Actually simpler approach: We'll do a nested loop over each component i
          pass

      for state_int in range(2**n):
            bits = [((state_int >> i) & 1) for i in range(n)]
            # For each component i, compare system function with i=0 vs i=1
            for i_comp in range(n):
                if bits[i_comp] == 0:
                    # We check system with i_comp=0 (current) vs i_comp=1 (flipped)
                    bits_zero = bits[:]
                    bits_one  = bits[:]
                    bits_zero[i_comp] = 0
                    bits_one[i_comp]  = 1

                    up_zero = system_up_bool(bits_zero)
                    up_one  = system_up_bool(bits_one)
                    # If flipping from 0->1 changes from fail->up => critical
                    if (not up_zero) and up_one:
                        B_counts[i_comp] += 1

                # Format output as a list with labels
      structural_birnbaum_index = [count / 2**(n - 1) for count in B_counts]
      return structural_birnbaum_index

def calculate_scenarios(n, use_paths, structures, p_syms, p_val):
    
    usage_list = []

    for i in range (1,n+1):
        result_i = brute_force_reliability(n, use_paths, structures, comp_index = i)

        scenario_expressions = {
            "system_up_comp_up":     result_i['system_up_comp_up_expr'],
            "system_up_comp_down":   result_i['system_up_comp_down_expr'],
            "system_down_comp_up":   result_i['system_down_comp_up_expr'],
            "system_down_comp_down": result_i['system_down_comp_down_expr']        
        }

        scenario_counts = {
            "system_up_comp_up":     result_i['system_up_comp_up_count'],
            "system_up_comp_down":   result_i['system_up_comp_down_count'],
            "system_down_comp_up":   result_i['system_down_comp_up_count'],
            "system_down_comp_down": result_i['system_down_comp_down_count']        
        }

        subs_dict = {p_syms[j]: p_val for j in range(n)}
        scenario_probs = {
            name: float(expr.subs(subs_dict).evalf())
            for name, expr in scenario_expressions.items()
        }

        usage_list.append({
            'comp_index': i,
            "scenario_probabilities":scenario_probs,
            'scenario_counts': scenario_counts
        })
    return usage_list

def compute_critical_states(n, use_paths, structures,comp_index=None):
    """
    Enumerate all 2^n states and determine:
      - num_system_critical_states = states where flipping ANY component changes system up<->down
      - num_comp_critical_states   = states where flipping comp_index changes system outcome (if provided)
    """
    # Step 1: define a helper that tells if the system is up for a given bit pattern
    def system_up_bool(bits):
        if use_paths:
            for path in structures:
                if all(bits[c-1] == 1 for c in path):
                    return True
            return False
        else:
            # minimal cuts => system fails if any cut is fully down
            for cut in structures:
                if all(bits[c-1] == 0 for c in cut):
                    return False
            return True

    # Step 2: if comp_index is None => compute the system-wide "critical states"
    if comp_index is None:
        # System-critical states = states where flipping ANY single component
        # changes system from down to up or up to down.
        # That means for a given bits pattern, if for ANY component i flipping i changes
        # the system outcome, that state is "system-critical."

        n_crit = 0
        for state_int in range(2**n):
            bits = [((state_int >> i) & 1) for i in range(n)]
            sys_val = system_up_bool(bits)
            
            # Check if flipping ANY component changes outcome
            flips_updown = False
            for i_comp in range(n):
                flipped_bits = bits[:]
                flipped_bits[i_comp] = 1 - flipped_bits[i_comp]
                flipped_val = system_up_bool(flipped_bits)
                if flipped_val != sys_val:
                    flips_updown = True
                    break
            if flips_updown:
                n_crit += 1
        
        return {
            'num_system_critical_states': n_crit
        }

    else:
        # Step 3: if comp_index is given => compute how many states are critical with respect to that comp
        # i.e. flipping that single comp changes system from fail to up or up to fail
        i = comp_index - 1  # to handle 1-based indexing vs 0-based lists
        n_crit = 0
        for state_int in range(2**n):
            bits = [((state_int >> b) & 1) for b in range(n)]
            sys_val = system_up_bool(bits)

            flipped_bits = bits[:]
            flipped_bits[i] = 1 - flipped_bits[i]
            flipped_val = system_up_bool(flipped_bits)
            if flipped_val != sys_val:
                n_crit += 1
        
        return {
            'num_comp_critical_states': n_crit
        }
    
def gather_critical_data(n, use_paths, structures):
    # get system-wide
    sys_res = compute_critical_states(n, use_paths, structures, comp_index=None)
    system_crit = sys_res['num_system_critical_states']

    # get per-component
    per_comp_list = []
    for i in range(1, n+1):
        c_res = compute_critical_states(n, use_paths, structures, comp_index=i)
        # c_res['num_comp_critical_states']
        per_comp_list.append({
            'comp_index': i,
            'comp_critical_states': c_res['num_comp_critical_states']
        })

    return {
        'system_critical_states': system_crit,
        'per_comp': per_comp_list
    }


# ------------------------------
# Main Calculation Function (Accepts Inputs)
# ------------------------------

def calculate_reliability(n, use_paths, structures, p_rob, l,comp_index=None):
    """
    Main function that accepts user parameters and returns results
    Returns: Dictionary with all computed values (serializable for JSON)
    """
    # Validate input structures
    for path in structures:
        if any(c < 1 or c > n for c in path):
            raise ValueError(f"Invalid component numbers in structures. Must be between 1 and {n}")

    # Core calculations
    result = brute_force_reliability(n, use_paths, structures)
    
    reliability_poly = result['reliability_poly']
    p_syms = result['p_syms']
    
    # Single-p calculations
    p, single_p_expr = collapse_to_p(reliability_poly, p_syms)
    val_p_rob = single_p_expr.subs({p: p_rob}).evalf()
    
    # Nu_s calculations
    nus_expr = compute_nus_derivative(single_p_expr)
    nus_expr2 = sympy.expand(nus_expr)
    nus_val = nus_expr.subs({p: p_rob}) * l
    
    # MTBF/MTDF calculations
    mut = val_p_rob / nus_val
    mdt = (1 - val_p_rob) / nus_val
    
    # Birnbaum calculations
    birnbaum_list = compute_birnbaum_multivariable(reliability_poly, p_syms)
    structural_birnbaum = compute_structural_birnbaum(n, use_paths, structures)

    # Add these additional calculations
    birnbaum_derivatives = []
    for i, expr in enumerate(birnbaum_list, start=1):
        single_p_birn = sympy.expand(expr.subs({p_syms[j]: p for j in range(n)}))
        numeric_val = single_p_birn.subs({p: p_rob}).evalf(5)
        
        # Calculate additional metrics for each component
        pcr = numeric_val * p_rob
        lb = pcr / val_p_rob
        bp = sympy.integrate(single_p_birn, (p, 0, 1))
        
        birnbaum_derivatives.append({
            'component': i,
            'I_B': str(single_p_birn),
            'I_B_value': float(numeric_val),
            'PCR': float(pcr),
            'Lb': float(lb),
            'BP': float(bp)
        })


    # Convert SymPy objects to strings and numerical values
    scenario_usage_all = calculate_scenarios(n,use_paths,structures,p_syms,p_rob)
    critical_data = gather_critical_data(n,use_paths,structures)

    results_dict = {
        'success_states': result['success_count'],
        'fail_states': result['fail_count'],
        'single_p_poly': str(single_p_expr),
        'reliability_value': float(val_p_rob),
        'nus_expr': str(nus_expr),
        'nus_value': float(nus_val),
        'mut': float(mut),
        'mdt': float(mdt),
        'birnbaum_derivatives': birnbaum_derivatives,
        'structural_birnbaum': [float(val) for val in structural_birnbaum]
    }

    results_dict["scenario_usage_all"] = scenario_usage_all
    results_dict['critical_data'] = critical_data

    return results_dict
# ------------------------------
# Demo Function (Original Example)
# ------------------------------

"""def demo_all():
    #Example usage with default parameters
    params = {
        'n': 6,
        'use_paths': True,
        'structures': [[1, 3], [1, 4], [2, 5], [1, 5, 6], [2, 3, 6], [2, 4, 6]],
        'p_rob': 0.9,
        'l': 0.0001
    
    }
    
    results = calculate_reliability(**params)
    
    # Print formatted results
    print(f"\n=== Demo Results ===")
    print(f"Reliability polynomial: {results['reliability_poly']}")
    print(f"Single-p polynomial: {results['single_p_poly']}")
    print(f"Reliability at p={params['p_rob']}: {results['reliability_value']:.4f}")
    print(f"MTBF: {results['mut']:.1f}, MTDF: {results['mdt']:.1f}")
    print("\nStructural Birnbaum Importance:")
    for i, val in enumerate(results['structural_birnbaum'], 1):
        print(f"  Component {i}: {val:.4f}")

# ------------------------------
# Execution Control
# ------------------------------

if __name__ == "__main__":
    # Run the demo example
    demo_all()
    
    # For real use, you would call calculate_reliability() directly with user parameters"""