"""
HTML visualization generator for piano hand prediction results.
Creates a styled HTML file with scrollable table showing all notes and highlighting incorrect predictions.
"""

import numpy as np


def pitch_to_name(pitch):
    """Convert MIDI pitch to note name."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = pitch // 12 - 1
    note = note_names[pitch % 12]
    return f"{note}{octave}"


def create_html_visualization(notes, predictions, ground_truth, output_path):
    """
    Create an HTML file with a scrollable table showing all notes and highlighting incorrect predictions.
    
    Args:
        notes: List of note dictionaries with keys: onset, offset, pitch, pitch_name, velocity
        predictions: numpy array of predicted hand labels (0=Right, 1=Left)
        ground_truth: numpy array of actual hand labels (0=Right, 1=Left)
        output_path: Path to save the HTML file
    """
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    incorrect = len(predictions) - correct
    accuracy = correct / len(predictions) * 100 if predictions.size > 0 else 0
    
    # Hand labels
    hand_labels = {0: 'Right', 1: 'Left'}
    
    # Calculate ranges for piano roll
    if notes:
        min_time = min(n['onset'] for n in notes)
        max_time = max(n['offset'] for n in notes)
        
        # Separate pitches by hand
        right_pitches = [notes[i]['pitch'] for i in range(len(notes)) if ground_truth[i] == 0]
        left_pitches = [notes[i]['pitch'] for i in range(len(notes)) if ground_truth[i] == 1]
        
        right_min_pitch = min(right_pitches) if right_pitches else 60
        right_max_pitch = max(right_pitches) if right_pitches else 84
        left_min_pitch = min(left_pitches) if left_pitches else 36
        left_max_pitch = max(left_pitches) if left_pitches else 60
        
        time_range = max_time - min_time if max_time > min_time else 1
        right_pitch_range = right_max_pitch - right_min_pitch if right_max_pitch > right_min_pitch else 24
        left_pitch_range = left_max_pitch - left_min_pitch if left_max_pitch > left_min_pitch else 24
    else:
        min_time = 0
        max_time = 1
        right_min_pitch = 60
        right_max_pitch = 84
        left_min_pitch = 36
        left_max_pitch = 60
        time_range = 1
        right_pitch_range = 24
        left_pitch_range = 24
    
    # Piano roll dimensions - fixed note width for consistent scrolling
    fixed_note_width = 10  # pixels per note (default, can be adjusted)
    svg_height_per_hand = 300
    svg_width = len(notes) * fixed_note_width  # Total width based on number of notes
    right_pitch_scale = svg_height_per_hand / right_pitch_range if right_pitch_range > 0 else svg_height_per_hand / 24
    left_pitch_scale = svg_height_per_hand / left_pitch_range if left_pitch_range > 0 else svg_height_per_hand / 24
    
    # Separate notes by hand (ground truth) and generate SVG rectangles
    right_hand_svg = []
    left_hand_svg = []
    
    for i, note in enumerate(notes):
        is_correct = predictions[i] == ground_truth[i]
        color = '#27ae60' if is_correct else '#e74c3c'  # Green for correct, red for incorrect
        
        # Fixed position and width - each note gets the same width
        x = i * fixed_note_width
        width = fixed_note_width - 1  # Small gap between notes
        
        if ground_truth[i] == 0:  # Right hand
            y = svg_height_per_hand - (note['pitch'] - right_min_pitch) * right_pitch_scale
            height = max(8, right_pitch_range / 30)  # Adaptive height
            rect = f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{color}" stroke="#333" stroke-width="0.5" opacity="0.8"><title>{note["pitch_name"]} ({note["onset"]:.2f}-{note["offset"]:.2f}s) - {"Correct" if is_correct else "Incorrect"}</title></rect>'
            right_hand_svg.append(rect)
        else:  # Left hand
            y = svg_height_per_hand - (note['pitch'] - left_min_pitch) * left_pitch_scale
            height = max(8, left_pitch_range / 30)  # Adaptive height
            rect = f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{color}" stroke="#333" stroke-width="0.5" opacity="0.8"><title>{note["pitch_name"]} ({note["onset"]:.2f}-{note["offset"]:.2f}s) - {"Correct" if is_correct else "Incorrect"}</title></rect>'
            left_hand_svg.append(rect)
    
    # Generate pitch labels and grid lines for y-axis (separate for each hand)
    right_pitch_labels_html = []
    right_grid_lines = []
    for pitch in range(int(right_min_pitch), int(right_max_pitch) + 1, 2):
        y_pos = svg_height_per_hand - (pitch - right_min_pitch) * right_pitch_scale
        note_name = pitch_to_name(pitch)
        # HTML label with positioning
        right_pitch_labels_html.append(f'<div class="piano-roll-y-label" style="top: {y_pos:.2f}px;">{note_name}</div>')
        # Grid line
        right_grid_lines.append(f'<line x1="0" y1="{y_pos:.2f}" x2="{svg_width}" y2="{y_pos:.2f}" stroke="#e0e0e0" stroke-width="1"/>')
    
    left_pitch_labels_html = []
    left_grid_lines = []
    for pitch in range(int(left_min_pitch), int(left_max_pitch) + 1, 2):
        y_pos = svg_height_per_hand - (pitch - left_min_pitch) * left_pitch_scale
        note_name = pitch_to_name(pitch)
        # HTML label with positioning
        left_pitch_labels_html.append(f'<div class="piano-roll-y-label" style="top: {y_pos:.2f}px;">{note_name}</div>')
        # Grid line
        left_grid_lines.append(f'<line x1="0" y1="{y_pos:.2f}" x2="{svg_width}" y2="{y_pos:.2f}" stroke="#e0e0e0" stroke-width="1"/>')
    
    # Generate note index labels (since we're using fixed width per note)
    num_labels = min(20, len(notes))  # Show up to 20 labels
    note_labels = []
    if len(notes) > 0:
        step = max(1, len(notes) // num_labels)
        for i in range(0, len(notes), step):
            x_pos = i * fixed_note_width + fixed_note_width / 2
            note_labels.append(f'<text x="{x_pos:.2f}" y="{svg_height_per_hand + 15}" font-size="9" fill="#666" text-anchor="middle">{i + 1}</text>')
            # Add tick mark
            note_labels.append(f'<line x1="{x_pos:.2f}" y1="{svg_height_per_hand}" x2="{x_pos:.2f}" y2="{svg_height_per_hand + 5}" stroke="#666" stroke-width="1"/>')
    
    # Generate table rows
    table_rows = []
    for i, note in enumerate(notes):
        is_correct = predictions[i] == ground_truth[i]
        predicted_hand = hand_labels[predictions[i]]
        actual_hand = hand_labels[ground_truth[i]]
        
        # Highlight entire row if incorrect
        row_class = 'incorrect-row' if not is_correct else ''
        
        table_rows.append(f"""
        <tr class="{row_class}" data-hand="{ground_truth[i]}" data-status="{1 if is_correct else 0}">
            <td>{i + 1}</td>
            <td>{note['onset']:.3f}</td>
            <td>{note['offset']:.3f}</td>
            <td>{note['pitch_name']}</td>
            <td>{note['pitch']}</td>
            <td>{note['velocity']}</td>
            <td class="hand-cell correct-hand">{actual_hand}</td>
            <td class="hand-cell {'correct-hand' if is_correct else 'incorrect-hand'}">{predicted_hand}</td>
            <td class="status-cell {'correct' if is_correct else 'incorrect'}">{'✓' if is_correct else '✗'}</td>
        </tr>
        """)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Piano Hand Prediction Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: white;
            color: #2c3e50;
            padding: 30px;
            text-align: center;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .stat-box {{
            background: #f8f9fa;
            padding: 15px 30px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        
        .stat-box .label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .stat-box .value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .table-container {{
            max-height: 70vh;
            overflow-y: auto;
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }}
        
        thead {{
            position: sticky;
            top: 0;
            z-index: 10;
            background: #2c3e50;
            color: white;
        }}
        
        th {{
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
            border-bottom: 3px solid #34495e;
        }}
        
        th.sortable {{
            cursor: pointer;
            user-select: none;
            position: relative;
            padding-right: 25px;
        }}
        
        th.sortable:hover {{
            background-color: #34495e;
        }}
        
        th.sortable::after {{
            content: ' ↕';
            position: absolute;
            right: 8px;
            opacity: 0.5;
            font-size: 0.8em;
        }}
        
        th.sortable.sort-asc::after {{
            content: ' ↑';
            opacity: 1;
        }}
        
        th.sortable.sort-desc::after {{
            content: ' ↓';
            opacity: 1;
        }}
        
        tbody tr {{
            border-bottom: 1px solid #e0e0e0;
            transition: background-color 0.2s ease;
        }}
        
        tbody tr:hover {{
            background-color: #f8f9fa;
        }}
        
        tbody tr.incorrect-row {{
            background-color: #ffebee;
            border-left: 4px solid #e74c3c;
        }}
        
        tbody tr.incorrect-row:hover {{
            background-color: #ffcdd2;
        }}
        
        td {{
            padding: 12px;
            color: #333;
        }}
        
        .hand-cell {{
            font-weight: 600;
            text-align: center;
        }}
        
        .hand-cell.correct-hand {{
            color: #27ae60;
        }}
        
        .hand-cell.incorrect-hand {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .status-cell {{
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .status-cell.correct {{
            color: #27ae60;
        }}
        
        .status-cell.incorrect {{
            color: #e74c3c;
        }}
        
        .note-number {{
            font-weight: 600;
            color: #7f8c8d;
        }}
        
        .pitch-name {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        /* Scrollbar styling */
        .table-container::-webkit-scrollbar {{
            width: 12px;
            height: 12px;
        }}
        
        .table-container::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        
        .table-container::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 6px;
        }}
        
        .table-container::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
        
        .legend {{
            padding: 20px 30px;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        
        .legend-color.correct {{
            background-color: #27ae60;
        }}
        
        .legend-color.incorrect {{
            background-color: #e74c3c;
        }}
        
        .legend-color.highlight {{
            background-color: #ffebee;
            border-left: 4px solid #e74c3c;
        }}
        
        .piano-roll-section {{
            padding: 30px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .piano-roll-title {{
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2c3e50;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .width-control {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.9em;
            color: #666;
            padding: 8px 15px;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        
        .width-control label {{
            font-weight: 500;
            white-space: nowrap;
        }}
        
        .width-control input[type="range"] {{
            width: 150px;
            height: 6px;
            cursor: pointer;
        }}
        
        .width-control span {{
            min-width: 50px;
            text-align: right;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .piano-roll-container {{
            background: #fafafa;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }}
        
        .piano-roll-label {{
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 10px;
            color: #34495e;
        }}
        
        .piano-roll-wrapper {{
            display: flex;
            position: relative;
        }}
        
        .piano-roll-y-axis {{
            position: sticky;
            left: 0;
            z-index: 5;
            background: white;
            padding-right: 10px;
            border-right: 2px solid #ddd;
        }}
        
        .piano-roll-y-labels {{
            height: {svg_height_per_hand + 20}px;
            position: relative;
        }}
        
        .piano-roll-y-label {{
            font-size: 10px;
            color: #333;
            text-align: right;
            padding-right: 8px;
            position: absolute;
            transform: translateY(-50%);
        }}
        
        .piano-roll-scrollable {{
            overflow-x: auto;
            overflow-y: hidden;
            flex: 1;
        }}
        
        .piano-roll-svg {{
            width: 100%;
            height: {svg_height_per_hand + 20}px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            transition: min-width 0.2s ease;
        }}
        
        .piano-roll-svg rect:hover {{
            opacity: 1;
            stroke-width: 1.5;
        }}
    </style>
    <script>
        // Synchronize scrolling between the two piano roll charts
        function syncScroll() {{
            const rightScroll = document.getElementById('right-hand-scroll');
            const leftScroll = document.getElementById('left-hand-scroll');
            
            let isScrolling = false;
            
            function handleScroll(scrollSource, scrollTarget) {{
                if (!isScrolling) {{
                    isScrolling = true;
                    scrollTarget.scrollLeft = scrollSource.scrollLeft;
                    setTimeout(() => {{ isScrolling = false; }}, 10);
                }}
            }}
            
            rightScroll.addEventListener('scroll', () => handleScroll(rightScroll, leftScroll));
            leftScroll.addEventListener('scroll', () => handleScroll(leftScroll, rightScroll));
        }}
        
        // Adjust chart width based on slider
        function adjustChartWidth() {{
            const slider = document.getElementById('width-slider');
            const widthValue = document.getElementById('width-value');
            const svgs = document.querySelectorAll('.piano-roll-svg');
            const numNotes = {len(notes)};
            const baseWidth = {svg_width};
            
            function updateWidth() {{
                const noteWidth = parseFloat(slider.value);
                const totalWidth = numNotes * noteWidth;
                
                widthValue.textContent = noteWidth.toFixed(1) + 'px';
                
                svgs.forEach(svg => {{
                    svg.style.minWidth = totalWidth + 'px';
                }});
            }}
            
            slider.addEventListener('input', updateWidth);
            updateWidth(); // Initial update
        }}
        
        // Table sorting functionality
        function initTableSorting() {{
            const table = document.querySelector('table');
            const tbody = table.querySelector('tbody');
            const sortableHeaders = document.querySelectorAll('th.sortable');
            
            let currentSort = {{
                column: null,
                direction: 'asc'
            }};
            
            sortableHeaders.forEach(header => {{
                header.addEventListener('click', () => {{
                    const sortType = header.getAttribute('data-sort');
                    const rows = Array.from(tbody.querySelectorAll('tr'));
                    
                    // Toggle sort direction if clicking the same column
                    if (currentSort.column === sortType) {{
                        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
                    }} else {{
                        currentSort.column = sortType;
                        currentSort.direction = 'asc';
                    }}
                    
                    // Remove sort classes from all headers
                    sortableHeaders.forEach(h => {{
                        h.classList.remove('sort-asc', 'sort-desc');
                    }});
                    
                    // Add sort class to current header
                    header.classList.add(currentSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
                    
                    // Sort rows
                    rows.sort((a, b) => {{
                        let aVal, bVal;
                        
                        if (sortType === 'hand') {{
                            // Sort by hand: 0=Right, 1=Left
                            aVal = parseInt(a.getAttribute('data-hand'));
                            bVal = parseInt(b.getAttribute('data-hand'));
                        }} else if (sortType === 'status') {{
                            // Sort by status: 0=Incorrect, 1=Correct
                            aVal = parseInt(a.getAttribute('data-status'));
                            bVal = parseInt(b.getAttribute('data-status'));
                        }}
                        
                        if (currentSort.direction === 'asc') {{
                            return aVal - bVal;
                        }} else {{
                            return bVal - aVal;
                        }}
                    }});
                    
                    // Re-append sorted rows
                    rows.forEach(row => tbody.appendChild(row));
                }});
            }});
        }}
        
        window.addEventListener('DOMContentLoaded', () => {{
            syncScroll();
            adjustChartWidth();
            initTableSorting();
        }});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎹 Piano Hand Prediction Results</h1>
            <div class="stats">
                <div class="stat-box">
                    <div class="label">Total Notes</div>
                    <div class="value">{len(notes)}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Correct</div>
                    <div class="value" style="color: #27ae60;">{correct}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Incorrect</div>
                    <div class="value" style="color: #e74c3c;">{incorrect}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Accuracy</div>
                    <div class="value">{accuracy:.2f}%</div>
                </div>
            </div>
        </div>
        
        <div class="piano-roll-section">
            <div class="piano-roll-title">
                <span>Piano Roll Visualization</span>
                <div class="width-control">
                    <label for="width-slider">Note Width:</label>
                    <input type="range" id="width-slider" min="5" max="30" step="1" value="{fixed_note_width}">
                    <span id="width-value">{fixed_note_width}px</span>
                </div>
            </div>
            <div class="piano-roll-container">
                <div class="piano-roll-label">Right Hand (Treble)</div>
                <div class="piano-roll-wrapper">
                    <div class="piano-roll-y-axis">
                        <div class="piano-roll-y-labels">
                            {''.join(right_pitch_labels_html)}
                        </div>
                    </div>
                    <div class="piano-roll-scrollable" id="right-hand-scroll">
                        <svg class="piano-roll-svg" viewBox="0 0 {svg_width} {svg_height_per_hand + 20}" preserveAspectRatio="none">
                            {''.join(right_grid_lines)}
                            {''.join(right_hand_svg)}
                            {''.join(note_labels)}
                        </svg>
                    </div>
                </div>
            </div>
            <div class="piano-roll-container">
                <div class="piano-roll-label">Left Hand (Bass)</div>
                <div class="piano-roll-wrapper">
                    <div class="piano-roll-y-axis">
                        <div class="piano-roll-y-labels">
                            {''.join(left_pitch_labels_html)}
                        </div>
                    </div>
                    <div class="piano-roll-scrollable" id="left-hand-scroll">
                        <svg class="piano-roll-svg" viewBox="0 0 {svg_width} {svg_height_per_hand + 20}" preserveAspectRatio="none">
                            {''.join(left_grid_lines)}
                            {''.join(left_hand_svg)}
                            {''.join(note_labels)}
                        </svg>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Onset (s)</th>
                        <th>Offset (s)</th>
                        <th>Note</th>
                        <th>Pitch</th>
                        <th>Velocity</th>
                        <th class="sortable" data-sort="hand">Actual Hand</th>
                        <th>Predicted Hand</th>
                        <th class="sortable" data-sort="status">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color correct"></div>
                <span>Correct Prediction</span>
            </div>
            <div class="legend-item">
                <div class="legend-color incorrect"></div>
                <span>Incorrect Prediction</span>
            </div>
            <div class="legend-item">
                <div class="legend-color highlight"></div>
                <span>Entire row highlighted when incorrect</span>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return accuracy, correct, incorrect

