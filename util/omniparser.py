from copy import deepcopy
from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
from util.output_schema import VISION_SCHEMA
from util.scene_graph import build_scene_graph
import torch
from PIL import Image
import io
import base64
from typing import Dict, List, Any


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(
            model_name=config['caption_model_name'],
            model_name_or_path=config['caption_model_path'],
            device=device,
        )
        print('Omniparser initialized!!!')

    def _normalize_bbox(self, bbox: List[float]) -> List[float]:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

    def _text_tokens(self, text: str) -> str:
        return (text or '').strip().lower()

    def classify_element(self, element: Dict[str, Any], application: str = None) -> str:
        text = self._text_tokens(element.get('content', ''))
        elem_type = element.get('type', 'icon')
        bbox = element.get('bbox', [0.0, 0.0, 0.0, 0.0])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        center_y = (bbox[1] + bbox[3]) / 2.0
        center_x = (bbox[0] + bbox[2]) / 2.0

        if 'notification' in text or 'alert' in text or 'warning' in text:
            return 'notification'

        if elem_type == 'text':
            if any(tag in text for tag in ['submit', 'send', 'save', 'cancel', 'ok', 'close', 'run', 'apply', 'upload', 'download']):
                return 'button'
            if any(tag in text for tag in ['tab', 'chatgpt', 'gmail', 'youtube', 'inbox', 'kaggle', 'notebook']):
                if center_y < 0.2:
                    return 'browser_tab'
                return 'tab'
            if any(tag in text for tag in ['search', 'enter', 'username', 'password', 'email', 'message', 'comment', 'query', 'prompt']):
                return 'textbox' if 'text' in text or 'search' in text else 'input'
            if any(tag in text for tag in ['link', 'www.', 'https://', 'http://']):
                return 'link'
            if width > 0.7 and height < 0.15:
                return 'toolbar'
            if width > 0.6 and height > 0.25:
                return 'panel'
            if center_y < 0.18 and width > 0.25:
                return 'tab'
            return 'panel'

        if elem_type == 'icon':
            if any(tag in text for tag in ['menu', 'hamburger', 'settings', 'gear', 'option', 'dots', 'more']):
                return 'menu'
            if any(tag in text for tag in ['close', 'maximize', 'minimize', 'restore', 'window']):
                return 'window'
            if any(tag in text for tag in ['search', 'zoom', 'share', 'download', 'upload']):
                return 'toolbar'
            if any(tag in text for tag in ['home', 'back', 'forward', 'refresh', 'bookmark']):
                return 'toolbar'
            if any(tag in text for tag in ['tab', 'browser', 'chrome', 'edge', 'firefox']):
                return 'browser_tab'
            if width < 0.08 and height < 0.08:
                return 'icon'
            if area > 0.2:
                return 'image'
            return 'icon'

        return 'panel'

    def infer_application(self, visible_text: List[str], elements: List[Dict[str, Any]]) -> str:
        text = ' '.join([self._text_tokens(t) for t in visible_text])
        if any(tag in text for tag in ['google chrome', 'chrome']):
            return 'Google Chrome'
        if 'chatgpt' in text:
            return 'ChatGPT'
        if 'kaggle' in text or 'notebook' in text:
            return 'Kaggle Notebook'
        if 'visual studio code' in text or 'vscode' in text or 'vs code' in text:
            return 'VSCode'
        if 'youtube' in text:
            return 'YouTube'
        if 'gmail' in text or 'inbox' in text or 'sent mail' in text:
            return 'Gmail'
        if any(tag in text for tag in ['file explorer', 'this pc', 'documents', 'downloads', 'desktop']):
            return 'File Explorer'
        if 'settings' in text or 'preferences' in text or 'system settings' in text:
            return 'Settings'
        if 'terminal' in text or 'bash' in text or 'zsh' in text or 'powershell' in text or 'cmd.exe' in text:
            return 'Terminal'
        if any(el.get('type') == 'icon' and 'tab' in self._text_tokens(el.get('content', '')) for el in elements):
            return 'Browser'
        return 'Browser'

    def summarize_screen(self, application: str, elements: List[Dict[str, Any]]) -> str:
        button_count = sum(1 for el in elements if el.get('type') == 'button')
        tab_count = sum(1 for el in elements if el.get('type') in ['tab', 'browser_tab'])
        icon_count = sum(1 for el in elements if el.get('type') == 'icon')
        text_count = sum(1 for el in elements if el.get('type') in ['textbox', 'input', 'link'])
        return (
            f"Detected {len(elements)} elements ({button_count} buttons, {tab_count} tabs, {icon_count} icons, "
            f"{text_count} interactive text items) in a {application} interface."
        )

    def build_regions(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        regions = []
        top_candidates = [el for el in elements if el['bbox'][1] < 0.15 and el['bbox'][2] - el['bbox'][0] > 0.3]
        left_candidates = [el for el in elements if el['bbox'][0] < 0.15 and el['bbox'][3] - el['bbox'][1] > 0.3]
        right_candidates = [el for el in elements if el['bbox'][2] > 0.85 and el['bbox'][3] - el['bbox'][1] > 0.3]
        if top_candidates:
            regions.append({'type': 'top_toolbar', 'bbox': [0.0, 0.0, 1.0, max(el['bbox'][3] for el in top_candidates)]})
        if left_candidates:
            regions.append({'type': 'left_sidebar', 'bbox': [0.0, 0.0, 0.2, max(el['bbox'][3] for el in left_candidates)]})
        if right_candidates:
            regions.append({'type': 'right_sidebar', 'bbox': [0.8, 0.0, 1.0, max(el['bbox'][3] for el in right_candidates)]})
        if not regions:
            regions.append({'type': 'main_panel', 'bbox': [0.0, 0.0, 1.0, 1.0]})
        return regions

    def element_clickable(self, element: Dict[str, Any]) -> bool:
        if element.get('type') in ['button', 'tab', 'browser_tab', 'link', 'menu']:
            return True
        if element.get('type') == 'icon' and element.get('interactivity', False):
            return True
        return element.get('interactivity', False)

    def element_confidence(self, element: Dict[str, Any]) -> float:
        base = 0.5
        if element.get('type') in ['button', 'tab', 'browser_tab', 'link', 'menu']:
            base = 0.95
        elif element.get('type') == 'icon':
            base = 0.85 if element.get('interactivity', False) else 0.55
        elif element.get('type') in ['textbox', 'input']:
            base = 0.9
        if element.get('content'):
            base = min(base + 0.05, 1.0)
        return float(round(base, 2))

    def possible_actions(self, application: str, elements: List[Dict[str, Any]]) -> List[str]:
        actions = []
        for el in elements:
            if not self.element_clickable(el):
                continue
            text = el.get('content', '') or el.get('type', '')
            label = text.strip().title() or el.get('type', '').replace('_', ' ').title()
            if el.get('type') == 'button':
                actions.append(f'Press {label}')
            elif el.get('type') in ['tab', 'browser_tab']:
                actions.append(f'Switch to {label}')
            elif el.get('type') == 'link':
                actions.append(f'Follow link {label}')
            elif el.get('type') == 'menu':
                actions.append(f'Open menu {label}')
            elif el.get('type') == 'icon':
                actions.append(f'Click icon {label}')
            elif el.get('type') in ['textbox', 'input']:
                actions.append(f'Focus input {label}')
        if 'ChatGPT' in application:
            actions.append('Open ChatGPT tab')
        if application == 'Google Chrome' or application == 'Browser':
            actions.extend(['Open new browser tab', 'Close browser tab'])
        if application == 'Kaggle Notebook':
            actions.extend(['Run notebook', 'Save notebook'])
        if application == 'Terminal':
            actions.extend(['Run command', 'Clear terminal'])
        if application == 'Gmail':
            actions.extend(['Compose email', 'Open inbox'])
        if application == 'YouTube':
            actions.extend(['Play video', 'Search video'])
        seen = set()
        cleaned = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                cleaned.append(action)
        return cleaned

    def estimate_user_focus(self, application: str, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        selected_tab = next((el for el in elements if el.get('type') == 'browser_tab'), None)
        if selected_tab:
            return {'type': 'tab', 'text': selected_tab.get('content', ''), 'bbox': selected_tab.get('bbox', [0.0, 0.0, 0.0, 0.0])}

        active_button = next((el for el in elements if el.get('type') == 'button' and any(k in self._text_tokens(el.get('content', '')) for k in ['run', 'send', 'submit', 'save', 'close'])), None)
        if active_button:
            return {'type': 'button', 'text': active_button.get('content', ''), 'bbox': active_button.get('bbox', [0.0, 0.0, 0.0, 0.0])}

        focus_input = next((el for el in elements if el.get('type') in ['textbox', 'input']), None)
        if focus_input:
            return {'type': 'textbox', 'text': focus_input.get('content', ''), 'bbox': focus_input.get('bbox', [0.0, 0.0, 0.0, 0.0])}

        sorted_elements = sorted(
            elements,
            key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]),
            reverse=True,
        )
        window_element = next((el for el in sorted_elements if el.get('type') in ['window', 'panel', 'browser_tab']), None)
        if window_element:
            return {
                'type': window_element.get('type', 'window'),
                'text': window_element.get('content', ''),
                'bbox': window_element.get('bbox', [0.0, 0.0, 0.0, 0.0]),
            }

        if elements:
            first = elements[0]
            return {'type': first.get('type', 'panel'), 'text': first.get('content', ''), 'bbox': first.get('bbox', [0.0, 0.0, 0.0, 0.0])}
        return None

    def parse(self, image_base64: str):
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print('image size:', image.size)

        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        (ocr_text, ocr_bbox), _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'text_threshold': 0.8},
            use_paddleocr=False,
        )

        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            self.som_model,
            BOX_TRESHOLD=self.config['BOX_TRESHOLD'],
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=ocr_text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128,
        )

        visible_text = [txt for txt in ocr_text if txt.strip()]
        application = self.infer_application(visible_text, parsed_content_list)

        elements = []
        for idx, raw in enumerate(parsed_content_list):
            element_type = self.classify_element(raw, application=application)
            clickable = self.element_clickable({'type': element_type, **raw})
            element = {
                'id': idx,
                'type': element_type,
                'text': raw.get('content', ''),
                'bbox': self._normalize_bbox(raw.get('bbox', [0.0, 0.0, 0.0, 0.0])),
                'clickable': clickable,
                'confidence': self.element_confidence({'type': element_type, **raw}),
                'interactivity': raw.get('interactivity', False),
                'source': raw.get('source', ''),
            }
            elements.append(element)

        buttons = [el for el in elements if el['type'] == 'button']
        tabs = [el for el in elements if el['type'] in ['tab', 'browser_tab']]
        icons = [el for el in elements if el['type'] == 'icon']
        ocr_entries = [
            {'text': txt, 'bbox': self._normalize_bbox(box), 'id': idx}
            for idx, (txt, box) in enumerate(zip(ocr_text, ocr_bbox))
        ]
        clickable_elements = [el for el in elements if el['clickable']]
        scene_graph = build_scene_graph(elements)
        regions = self.build_regions(elements)
        possible_actions = self.possible_actions(application, elements)
        user_focus = self.estimate_user_focus(application, elements)
        screen_summary = self.summarize_screen(application, elements)

        result = {
            'application': application,
            'screen_summary': screen_summary,
            'elements': elements,
            'buttons': buttons,
            'tabs': tabs,
            'icons': icons,
            'ocr': ocr_entries,
            'regions': regions,
            'scene_graph': scene_graph,
            'clickable_elements': clickable_elements,
            'possible_actions': possible_actions,
            'visible_text': visible_text,
            'user_focus': user_focus,
        }

        return dino_labled_img, result
