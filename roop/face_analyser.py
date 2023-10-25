import threading
import insightface
import numpy
import roop.globals
from roop.typing import Frame, Face

THREAD_LOCK = threading.Lock()
FACE_ANALYSER = None

def get_face_analyser() -> Any:
    """
    Get or initialize the face analyzer.

    This function returns the global FACE_ANALYSER instance if it exists, or initializes it if it's None.

    Returns:
        Any: The face analyzer instance.
    """
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER

def clear_face_analyser() -> Any:
    """
    Clear the global face analyzer instance.
    """
    global FACE_ANALYSER
    FACE_ANALYSER = None

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    """
    Detect one face in a given frame.

    Args:
        frame (Frame): The input frame.
        position (int): The position of the detected face.

    Returns:
        Optional[Face]: The detected face or None if no face is found.
    """
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """
    Detect multiple faces in a given frame.

    Args:
        frame (Frame): The input frame.

    Returns:
        Optional[List[Face]]: A list of detected faces or None if no faces are found.
    """
    try:
        return get_face_analyser().get(frame)
    except Exception as e:
        # Improve error handling by logging the error
        import logging
        logging.error(f"Error in get_many_faces: {str(e)}")
        return None

def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    """
    Find a similar face in the given frame.

    Args:
        frame (Frame): The input frame.
        reference_face (Face): The reference face to compare against.

    Returns:
        Optional[Face]: A similar face or None if no similar face is found.
    """
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
