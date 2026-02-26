
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "description": "A fungal disease caused by Venturia inaequalis that affects apple trees, causing dark, scaly lesions on leaves and fruit.",
        "treatment": "Apply fungicides such as captan or sulfur. Remove and destroy fallen leaves to reduce overwintering spores."
    },
    "Apple___Black_rot": {
        "description": "A fungal disease caused by Botryosphaeria obtusa that causes leaf spots, fruit rot, and cankers on branches.",
        "treatment": "Prune out dead wood and cankers. Apply fungicides like captan or thiophanate-methyl during the growing season."
    },
    "Apple___Cedar_apple_rust": {
        "description": "A fungal disease that requires both apple and cedar trees to complete its life cycle, causing bright orange spots on leaves.",
        "treatment": "Remove nearby cedar trees if possible. Apply preventive fungicides in early spring."
    },
    "Apple___healthy": {
        "description": "The apple leaf is healthy and shows no signs of disease.",
        "treatment": "Continue regular monitoring and good cultural practices like proper watering and fertilization."
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Caused by the fungus Puccinia sorghi, appearing as reddish-brown pustules on both leaf surfaces.",
        "treatment": "Plant resistant hybrids. Fungicides can be used if infection is severe and detected early."
    },
    "Corn_(maize)___healthy": {
        "description": "The corn leaf is healthy and shows no signs of disease.",
        "treatment": "Maintain proper spacing and soil health to prevent future issues."
    },
    "Grape___Black_rot": {
        "description": "One of the most serious fungal diseases of grapes, causing small brown spots on leaves and shriveling of berries into mummies.",
        "treatment": "Ensure good air circulation through pruning. Apply fungicides from bud break through bloom."
    },
    "Grape___healthy": {
        "description": "The grape leaf is healthy and shows no signs of disease.",
        "treatment": "Monitor for pests and ensure proper trellising for air flow."
    },
    "Potato___Early_blight": {
        "description": "Caused by the fungus Alternaria solani, it produces dark spots with concentric rings (target-like appearance) on older leaves.",
        "treatment": "Use certified disease-free seeds. Apply copper-based fungicides and practice crop rotation."
    },
    "Potato___Late_blight": {
        "description": "A devastating disease caused by the oomycete Phytophthora infestans, responsible for the Irish Potato Famine.",
        "treatment": "Destroy infected plants immediately. Apply systemic fungicides and avoid overhead irrigation."
    },
    "Potato___healthy": {
        "description": "The potato leaf is healthy and shows no signs of disease.",
        "treatment": "Practice crop rotation and avoid planting near other solanaceous crops."
    },
    "Tomato___Early_blight": {
        "description": "Similar to potato early blight, it causes target-like spots on leaves, often leading to defoliation.",
        "treatment": "Remove lower leaves to prevent soil-borne spores from splashing onto foliage. Use fungicides if necessary."
    },
    "Tomato___Late_blight": {
        "description": "Causes rapidly expanding dark spots on leaves and stems, with white fungal growth on the underside in humid conditions.",
        "treatment": "Improve air circulation. Use resistant varieties and apply fungicides early in the season."
    },
    "Tomato___healthy": {
        "description": "The tomato leaf is healthy and shows no signs of disease.",
        "treatment": "Ensure consistent watering and use mulch to prevent soil splashing."
    }
}

def get_disease_info(disease_name):
    return DISEASE_INFO.get(disease_name, {
        "description": "Detailed information for this specific plant disease is currently being updated in our database.",
        "treatment": "General recommendation: Isolate the affected plant, remove symptomatic leaves, and consult a local agricultural expert."
    })
