import pandas as pd
from typing import Dict, Any, Optional

class ICTStrategy:
    def __init__(self, algorithm_version: str = "v4"):
        self.version = algorithm_version

    def generate_signal(self, current_data: Dict[str, Any], htf_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generates a trading signal based on the specified algorithm version.

        current_data: Dictionary containing the latest predicted/calculated ICT features for the base timeframe.
        htf_data: Dictionary containing HTF features (e.g. HTF Liquidity Grabs, POI).
        """
        if self.version == "v4":
            return self._v4_algorithm(current_data, htf_data)
        elif self.version == "v3":
            return self._v3_algorithm(current_data, htf_data)
        else:
            raise ValueError(f"Unknown algorithm version: {self.version}")

    def _v4_algorithm(self, current: Dict[str, Any], htf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        V4 5-Tier Confirmation System
        Mandatory: MSS (Tier 1), Risk-Reward Ratio >= 1:3.0 (Tier 5 - validated at execution)
        Optional: HTF Liquidity Grab, POI, FVG
        Requires: Both mandatory + at least 1 optional (3/5 total).
        """
        process_log = []

        # Determine Direction from MSS
        direction = 0
        close_price = current.get('close', 0.0)
        swing_high = current.get('recent_swing_high', 0.0)
        swing_low = current.get('recent_swing_low', 0.0)

        process_log.append(f"Checking for Mandatory Market Structure Shift (MSS)...")
        if current.get('mss') == 1:
            direction = 1 # Bullish
            if close_price > swing_high:
                mss_reason = f"Detected (Bullish MSS) - Close Price {close_price:.4f} broke above recent Swing High {swing_high:.4f}."
            else:
                mss_reason = f"Detected (Bullish MSS) - Triggered by AI Prediction Override (Math: {close_price:.4f} did not break {swing_high:.4f})."
            process_log.append(mss_reason)
        elif current.get('mss') == -1:
            direction = -1 # Bearish
            if close_price < swing_low:
                mss_reason = f"Detected (Bearish MSS) - Close Price {close_price:.4f} broke below recent Swing Low {swing_low:.4f}."
            else:
                mss_reason = f"Detected (Bearish MSS) - Triggered by AI Prediction Override (Math: {close_price:.4f} did not break {swing_low:.4f})."
            process_log.append(mss_reason)
        else:
            process_log.append("Not Detected. Mandatory MSS missing.")

        if direction == 0:
            return {"action": None, "thought_process": process_log}

        # Check optional tiers
        confirmations = 0
        reasons = [mss_reason]

        # 1. FVG presence in the same direction
        process_log.append("Checking for Fair Value Gap (FVG)...")
        if current.get('fvg') == direction:
            confirmations += 1
            fvg_upper = current.get('fvg_upper', 0.0)
            fvg_lower = current.get('fvg_lower', 0.0)
            reason = f"Optional: {'Bullish' if direction == 1 else 'Bearish'} Fair Value Gap (FVG) aligned [Zone: {fvg_lower:.4f} - {fvg_upper:.4f}]."
            reasons.append(reason)
            process_log.append(f"Detected. {reason}")
        else:
            process_log.append("Not Detected.")

        # 2. HTF Liquidity Grab in the same direction
        process_log.append("Checking for HTF Liquidity Grab...")
        if htf.get('liquidity_grab') == direction:
            confirmations += 1
            reason = f"Optional: HTF {'Sell-side' if direction == 1 else 'Buy-side'} Liquidity Grab (spike and reverse) aligned against structural liquidity."
            reasons.append(reason)
            process_log.append(f"Detected. {reason}")
        else:
            process_log.append("Not Detected.")

        # 3. POI / Order Block in the same direction
        process_log.append("Checking for Order Block (POI)...")
        if current.get('ob') == direction or htf.get('ob') == direction:
            confirmations += 1
            ob_upper = current.get('ob_upper', 0.0)
            ob_lower = current.get('ob_lower', 0.0)
            ob_htf_upper = htf.get('ob_upper', 0.0)
            ob_htf_lower = htf.get('ob_lower', 0.0)

            # Use current OB range if it triggered, else use the HTF OB range
            upper = ob_upper if current.get('ob') == direction else ob_htf_upper
            lower = ob_lower if current.get('ob') == direction else ob_htf_lower

            reason = f"Optional: {'Bullish' if direction == 1 else 'Bearish'} Order Block aligned [Zone: {lower:.4f} - {upper:.4f}]."
            reasons.append(reason)
            process_log.append(f"Detected. {reason}")
        else:
            process_log.append("Not Detected.")

        process_log.append(f"Result: {confirmations} optional confirmations found. Minimum 1 required.")

        # Must have at least 1 optional confirmation
        if confirmations >= 1:
            process_log.append("Valid Setup Found! Triggering Trade Execution.")
            return {
                "action": "BUY" if direction == 1 else "SELL",
                "confidence": 0.7 + (0.1 * confirmations), # pseudo-confidence score
                "version": "v4",
                "tier_confirmations": confirmations + 2, # +2 for MSS and R:R
                "reasons": reasons,
                "thought_process": process_log
            }

        process_log.append("Setup invalid. Skipping trade.")
        return {"action": None, "thought_process": process_log}

    def _v3_algorithm(self, current: Dict[str, Any], htf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        V3 Algorithm (Production 75% Win Rate)
        Mandatory: MSS
        Requires Risk-Reward >= 1:1.2
        Confidence Threshold: 62%
        Requires 3 of 5 overall confirmations.
        """
        process_log = []
        direction = current.get('mss', 0)
        close_price = current.get('close', 0.0)
        swing_high = current.get('recent_swing_high', 0.0)
        swing_low = current.get('recent_swing_low', 0.0)

        process_log.append(f"Checking for Mandatory Market Structure Shift (MSS)...")
        if direction == 0:
            process_log.append("Not Detected. Mandatory MSS missing.")
            return {"action": None, "thought_process": process_log}

        if direction == 1:
            if close_price > swing_high:
                mss_reason = f"Detected (Bullish MSS) - Close Price {close_price:.4f} broke above recent Swing High {swing_high:.4f}."
            else:
                mss_reason = f"Detected (Bullish MSS) - Triggered by AI Prediction Override (Math: {close_price:.4f} did not break {swing_high:.4f})."
        else:
            if close_price < swing_low:
                mss_reason = f"Detected (Bearish MSS) - Close Price {close_price:.4f} broke below recent Swing Low {swing_low:.4f}."
            else:
                mss_reason = f"Detected (Bearish MSS) - Triggered by AI Prediction Override (Math: {close_price:.4f} did not break {swing_low:.4f})."

        process_log.append(mss_reason)

        confirmations = 0
        reasons = [mss_reason]

        process_log.append("Checking for Fair Value Gap (FVG)...")
        if current.get('fvg') == direction:
            confirmations += 1
            reason = f"Fair Value Gap near {close_price:.4f}"
            reasons.append(reason)
            process_log.append(f"Detected. {reason}")
        else:
            process_log.append("Not Detected.")

        process_log.append("Checking for Order Block...")
        if current.get('ob') == direction:
            confirmations += 1
            reason = "Institutional Order Block"
            reasons.append(reason)
            process_log.append(f"Detected. {reason}")
        else:
            process_log.append("Not Detected.")

        process_log.append("Checking for Liquidity Grab...")
        if current.get('liquidity_grab') == direction:
            confirmations += 1
            reason = "Liquidity Grab Spike/Reversal"
            reasons.append(reason)
            process_log.append(f"Detected. {reason}")
        else:
            process_log.append("Not Detected.")

        process_log.append("Checking for Premium/Discount Alignment...")
        if current.get('premium_discount') == (1 if direction == 1 else -1):
            confirmations += 1
            reason = "Optimal Premium/Discount valuation territory"
            reasons.append(reason)
            process_log.append(f"Detected. {reason}")
        else:
            process_log.append("Not Detected.")

        process_log.append(f"Result: {confirmations} optional confirmations found. Minimum 2 required.")

        if confirmations >= 2: # MSS (1) + 2 others = 3/5
            process_log.append("Valid Setup Found! Triggering Trade Execution.")
            return {
                "action": "BUY" if direction == 1 else "SELL",
                "confidence": 0.62 + (0.05 * confirmations),
                "version": "v3",
                "tier_confirmations": confirmations + 1,
                "reasons": reasons,
                "thought_process": process_log
            }

        process_log.append("Setup invalid. Skipping trade.")
        return {"action": None, "thought_process": process_log}
